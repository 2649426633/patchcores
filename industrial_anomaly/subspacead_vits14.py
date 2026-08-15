from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SUBSPACEAD_SRC = REPO_ROOT / "third_party" / "subspacead" / "src"
for path in (REPO_ROOT, SUBSPACEAD_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.subspacead import DINOv2SSubspaceConfig, DINOv2SSubspaceExtractor
from subspacead.core.pca import PCAModel
from subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from subspacead.utils.common import min_max_norm, topk_mean


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PCA_EV = 0.99
DEFAULT_IMAGE_SIZE = 448
DEFAULT_BLOCKS = (7, 8)
DEFAULT_AUG_COUNT = 30
TOP_FRACTION = 0.01


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS],
        key=lambda p: p.name.lower(),
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {path}")
    return images


def default_model_dir(product: str) -> Path:
    return HERE / "products" / product / "models" / "subspacead_vits14"


def make_extractor(
    *,
    device: str | None,
    repo_dir: str,
    weights_path: str,
    image_size: int = DEFAULT_IMAGE_SIZE,
    block_indices: tuple[int, ...] = DEFAULT_BLOCKS,
) -> DINOv2SSubspaceExtractor:
    cfg = DINOv2SSubspaceConfig(
        image_size=int(image_size),
        block_indices=tuple(int(v) for v in block_indices),
        repo_dir=repo_dir,
        weights_path=weights_path,
    )
    extractor = DINOv2SSubspaceExtractor(device=device, config=cfg)
    extractor.load()
    return extractor


def save_pca(model_dir: Path, pca: dict, config: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_dir / "pca_model.npz",
        mu=np.asarray(pca["mu"], dtype=np.float64),
        components=np.asarray(pca["components"], dtype=np.float64),
        eigvals=np.asarray(pca["eigvals"], dtype=np.float64),
        sqrt_eig=np.asarray(pca["sqrt_eig"], dtype=np.float64),
        k=np.asarray([int(pca["k"])], dtype=np.int64),
        eps=np.asarray([float(pca["eps"])], dtype=np.float64),
    )
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_pca(model_dir: Path) -> tuple[dict, dict]:
    config_path = model_dir / "config.json"
    pca_path = model_dir / "pca_model.npz"
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not pca_path.exists():
        raise FileNotFoundError(pca_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    data = np.load(pca_path, allow_pickle=False)
    k = int(np.asarray(data["k"]).reshape(-1)[0])
    eps = float(np.asarray(data["eps"]).reshape(-1)[0])
    eigvals = np.asarray(data["eigvals"], dtype=np.float64)
    pca = {
        "mu": np.asarray(data["mu"], dtype=np.float64),
        "components": np.asarray(data["components"], dtype=np.float64),
        "eigvals": eigvals,
        "sqrt_eig": np.asarray(data["sqrt_eig"], dtype=np.float64),
        "k": k,
        "eps": eps,
        "whiten": False,
        "cov_Z_inv": np.diag(1.0 / (eigvals + eps)),
    }
    return pca, config


def fit_command(args: argparse.Namespace) -> None:
    normal_dir = resolve_path(args.normal_dir)
    model_dir = resolve_path(args.model_dir) if args.model_dir else default_model_dir(args.product)
    normal_images = collect_images(normal_dir)
    if len(normal_images) < args.shots:
        raise RuntimeError(
            f"Requested {args.shots}-shot but only {len(normal_images)} normal images exist"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    selected = normal_images.copy()
    random.shuffle(selected)
    selected = selected[: args.shots]

    # Match the official SubspaceAD DINOv2-S backbone ablation: 448 px,
    # equivalent middle layers -4,-5, mean aggregation, PCA EV 0.99.
    # Rotation angles are sampled once and reused in both PCA passes so the
    # streaming mean and covariance see exactly the same augmented views.
    angle_plan: dict[str, list[float | None]] = {}
    for path in selected:
        angles: list[float | None] = [None]
        angles.extend(random.uniform(0.0, 345.0) for _ in range(args.aug_count))
        angle_plan[str(path)] = angles

    print("========== SubspaceAD DINOv2-S FIT ==========")
    print(f"product:       {args.product}")
    print(f"normal dir:    {normal_dir}")
    print(f"shots:         {args.shots}")
    print(f"augmentations: {args.aug_count} rotations / image")
    print(f"resolution:    {DEFAULT_IMAGE_SIZE}")
    print(f"blocks:        {DEFAULT_BLOCKS}  (HF equivalent: -5,-4)")
    print(f"PCA EV:        {PCA_EV}")
    print(f"model dir:     {model_dir}")
    for i, path in enumerate(selected, 1):
        print(f"normal[{i}]:    {path.name}")

    extractor = make_extractor(
        device=args.device,
        repo_dir=args.dinov2_repo,
        weights_path=args.weights,
    )

    probe = extractor.extract_tokens([selected[0]])
    _, h_p, w_p, feature_dim = probe.shape
    tokens_per_image = h_p * w_p
    total_views = args.shots * (1 + args.aug_count)
    total_tokens = total_views * tokens_per_image
    num_batches = total_views
    print(
        f"feature_dim={feature_dim}, grid={h_p}x{w_p}, "
        f"views={total_views}, PCA tokens={total_tokens}"
    )

    def feature_generator():
        for path in selected:
            base = Image.open(path).convert("RGB")
            for angle in angle_plan[str(path)]:
                view = base if angle is None else TF.rotate(base, angle=float(angle))
                tokens = extractor.extract_tokens([view])
                yield tokens.reshape(-1, feature_dim)

    print("Fitting PCA (two streaming passes)...")
    pca_model = PCAModel(k=None, ev=PCA_EV, whiten=False)
    pca = pca_model.fit(
        feature_generator,
        feature_dim=feature_dim,
        total_tokens=total_tokens,
        num_batches=num_batches,
    )

    config = {
        "method": "SubspaceAD",
        "variant": "industrial_dinov2_vits14",
        "backbone": "dinov2_vits14",
        "source_weights": args.weights,
        "dinov2_repo": args.dinov2_repo,
        "feature_dim": feature_dim,
        "patch_size": 14,
        "image_size": DEFAULT_IMAGE_SIZE,
        "patch_grid": [h_p, w_p],
        "local_block_indices": list(DEFAULT_BLOCKS),
        "hf_equivalent_layers": [-5, -4],
        "aggregation": "mean",
        "pca_explained_variance": PCA_EV,
        "pca_components": int(pca["k"]),
        "augmentation": "rotation_0_to_345_degrees",
        "augmentation_count": int(args.aug_count),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "image_score": "mean_top_1_percent",
        "gaussian_sigma": 4.0,
        "selected_normal_images": [str(p) for p in selected],
        "note": "DINOv2-S setting follows the official SubspaceAD backbone-ablation configuration; this is not the Giant main-result configuration.",
    }
    save_pca(model_dir, pca, config)

    print("========== FIT DONE ==========")
    print(f"PCA components: {pca['k']}")
    print(f"saved: {model_dir / 'pca_model.npz'}")
    print(f"saved: {model_dir / 'config.json'}")


def _bbox_from_map(
    anomaly_map: np.ndarray, relative_threshold: float
) -> tuple[int, int, int, int] | None:
    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    mask = (normalized >= float(relative_threshold)).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def _save_result_visuals(
    image_path: Path,
    anomaly_map: np.ndarray,
    output_dir: Path,
    bbox_map: tuple[int, int, int, int] | None,
    image_score: float,
) -> tuple[Path, Path, Path, list[int] | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "anomaly_map.npy"
    np.save(raw_path, anomaly_map.astype(np.float32))

    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    heat_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_path = output_dir / "heatmap.jpg"
    cv2.imwrite(str(heat_path), heat)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")
    oh, ow = original.shape[:2]
    mh, mw = anomaly_map.shape[:2]
    heat_full = cv2.resize(heat, (ow, oh), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(original, 0.58, heat_full, 0.42, 0.0)

    bbox_original = None
    if bbox_map is not None:
        x1, y1, x2, y2 = bbox_map
        bx1 = max(0, min(ow - 1, int(round(x1 * ow / mw))))
        by1 = max(0, min(oh - 1, int(round(y1 * oh / mh))))
        bx2 = max(bx1 + 1, min(ow, int(round(x2 * ow / mw))))
        by2 = max(by1 + 1, min(oh, int(round(y2 * oh / mh))))
        bbox_original = [bx1, by1, bx2, by2]
        thickness = max(2, int(round(min(ow, oh) / 400.0)))
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 255), thickness)

    cv2.putText(
        overlay,
        f"SubspaceAD-S score={image_score:.4f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, min(1.2, min(ow, oh) / 1400.0)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay_path = output_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    return raw_path, heat_path, overlay_path, bbox_original


def inspect_command(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    images = collect_images(input_path)
    model_dir = resolve_path(args.model_dir) if args.model_dir else default_model_dir(args.product)
    pca, config = load_pca(model_dir)

    repo_dir = args.dinov2_repo or str(config.get("dinov2_repo", "third_party/dinov2"))
    weights = args.weights or str(config.get("source_weights", "weights/dinov2_vits14_pretrain.pth"))
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    blocks = tuple(int(v) for v in config.get("local_block_indices", DEFAULT_BLOCKS))

    extractor = make_extractor(
        device=args.device,
        repo_dir=repo_dir,
        weights_path=weights,
        image_size=image_size,
        block_indices=blocks,
    )

    output_root = (
        resolve_path(args.output_dir)
        if args.output_dir
        else HERE / "outputs" / args.product / "subspacead_vits14" / input_path.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, image_path in enumerate(images, 1):
        if extractor.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        tokens = extractor.extract_tokens([image_path])
        b, h_p, w_p, feature_dim = tokens.shape
        if b != 1:
            raise RuntimeError(f"Unexpected batch size: {b}")
        scores = calculate_anomaly_scores(
            tokens.reshape(h_p * w_p, feature_dim),
            pca,
            method="reconstruction",
            drop_k=0,
        )
        anomaly_low = scores.reshape(h_p, w_p)
        anomaly_map = post_process_map(anomaly_low, image_size)
        image_score = float(topk_mean(anomaly_map, frac=TOP_FRACTION))

        if extractor.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        bbox_map = _bbox_from_map(anomaly_map, args.bbox_relative_threshold)
        image_output = output_root if len(images) == 1 else output_root / image_path.stem
        raw_path, heat_path, overlay_path, bbox_original = _save_result_visuals(
            image_path,
            anomaly_map,
            image_output,
            bbox_map,
            image_score,
        )

        result = {
            "image": str(image_path),
            "method": "SubspaceAD",
            "variant": "industrial_dinov2_vits14",
            "image_score_top_1pct": image_score,
            "patch_score_min": float(np.min(scores)),
            "patch_score_max": float(np.max(scores)),
            "bbox_relative_threshold": float(args.bbox_relative_threshold),
            "bbox_original_image": bbox_original,
            "inference_ms": inference_ms,
            "device": str(extractor.device),
            "pass_ng": "UNCALIBRATED",
            "raw_map": str(raw_path),
            "heatmap": str(heat_path),
            "overlay": str(overlay_path),
        }
        (image_output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(result)
        print(
            f"[{index}/{len(images)}] {image_path.name} "
            f"score={image_score:.6f} bbox={bbox_original} "
            f"time={inference_ms:.1f} ms"
        )

    if len(rows) > 1:
        (output_root / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Output: {output_root.resolve()}")
    print("PASS/NG remains UNCALIBRATED; bbox threshold is localization-only.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Industrial SubspaceAD using the existing local DINOv2-S/14 weights."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit PCA from normal images")
    fit.add_argument("--product", required=True)
    fit.add_argument("--normal-dir", required=True)
    fit.add_argument("--shots", type=int, choices=[1, 2, 4], default=4)
    fit.add_argument("--aug-count", type=int, default=DEFAULT_AUG_COUNT)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default=None)
    fit.add_argument("--model-dir", default=None)
    fit.add_argument("--dinov2-repo", default="third_party/dinov2")
    fit.add_argument("--weights", default="weights/dinov2_vits14_pretrain.pth")
    fit.set_defaults(func=fit_command)

    inspect = sub.add_parser("inspect", help="Inspect one image or a folder")
    inspect.add_argument("--product", required=True)
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--device", default=None)
    inspect.add_argument("--model-dir", default=None)
    inspect.add_argument("--output-dir", default=None)
    inspect.add_argument("--dinov2-repo", default=None)
    inspect.add_argument("--weights", default=None)
    inspect.add_argument(
        "--bbox-relative-threshold",
        type=float,
        default=0.75,
        help="Relative heatmap threshold for localization only; not a PASS/NG threshold.",
    )
    inspect.set_defaults(func=inspect_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "aug_count", 0) < 0:
        parser.error("--aug-count must be >= 0")
    if hasattr(args, "bbox_relative_threshold") and not (
        0.0 < args.bbox_relative_threshold <= 1.0
    ):
        parser.error("--bbox-relative-threshold must be in (0,1]")
    args.func(args)


if __name__ == "__main__":
    main()
