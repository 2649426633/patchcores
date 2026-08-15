from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SUBSPACEAD_SRC = REPO_ROOT / "third_party" / "subspacead" / "src"
if str(SUBSPACEAD_SRC) not in sys.path:
    sys.path.insert(0, str(SUBSPACEAD_SRC))

from subspacead.core.extractor import FeatureExtractor
from subspacead.core.pca import PCAModel
from subspacead.data.transforms import get_augmentation_transform
from subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from subspacead.utils.common import min_max_norm, topk_mean


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_RES = 672
LAYERS = [-12, -13, -14, -15, -16, -17, -18]
AGG_METHOD = "mean"
PCA_EV = 0.99
AUGMENTATION = ["rotate"]
TOP_FRACTION = 0.01
DEFAULT_MODEL_CKPT = "weights/subspacead/dinov2-with-registers-giant"


def log(message: str) -> None:
    print(message, flush=True)


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
    return HERE / "products" / product / "models" / "subspacead_giant"


def default_output_root(product: str) -> Path:
    return HERE / "outputs" / product / "subspacead_giant"


def resolve_model_ckpt(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate.resolve())
    local = (REPO_ROOT / candidate).resolve()
    if local.exists():
        return str(local)
    return value


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
    pca_path = model_dir / "pca_model.npz"
    config_path = model_dir / "config.json"
    if not pca_path.exists():
        raise FileNotFoundError(
            f"PCA model not found: {pca_path}\nRun the 'fit' command first."
        )
    if not config_path.exists():
        raise FileNotFoundError(config_path)

    data = np.load(pca_path, allow_pickle=False)
    eigvals = np.asarray(data["eigvals"], dtype=np.float64)
    eps = float(np.asarray(data["eps"]).reshape(-1)[0])
    pca = {
        "mu": np.asarray(data["mu"], dtype=np.float64),
        "components": np.asarray(data["components"], dtype=np.float64),
        "eigvals": eigvals,
        "sqrt_eig": np.asarray(data["sqrt_eig"], dtype=np.float64),
        "k": int(np.asarray(data["k"]).reshape(-1)[0]),
        "eps": eps,
        "whiten": False,
        "cov_Z_inv": np.diag(1.0 / (eigvals + eps)),
    }
    config = json.loads(config_path.read_text(encoding="utf-8"))
    return pca, config


def extract_tokens(extractor: FeatureExtractor, image: Image.Image):
    return extractor.extract_tokens(
        [image],
        IMAGE_RES,
        LAYERS,
        AGG_METHOD,
        [],
        False,
        use_clahe=False,
        dino_saliency_layer=0,
    )


def fit_command(args: argparse.Namespace) -> None:
    normal_dir = resolve_path(args.normal_dir)
    model_dir = resolve_path(args.model_dir) if args.model_dir else default_model_dir(args.product)
    model_ckpt = resolve_model_ckpt(args.model_ckpt)
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

    log("========== SubspaceAD Giant FIT ==========")
    log(f"product:       {args.product}")
    log(f"normal dir:    {normal_dir}")
    log(f"shots:         {args.shots}")
    log(f"augmentations: {args.aug_count} rotations / normal image")
    log(f"model:         {model_ckpt}")
    log(f"resolution:    {IMAGE_RES}")
    log(f"layers:        {LAYERS}")
    log(f"aggregation:   {AGG_METHOD}")
    log(f"PCA EV:        {PCA_EV}")
    log(f"model dir:     {model_dir}")
    for i, path in enumerate(selected, 1):
        log(f"normal[{i}]:    {path.name}")

    log("[1/3] Loading official SubspaceAD FeatureExtractor...")
    extractor = FeatureExtractor(model_ckpt)

    first = Image.open(selected[0]).convert("RGB")
    probe, (h_p, w_p), _ = extract_tokens(extractor, first)
    feature_dim = int(probe.shape[-1])
    tokens_per_view = int(h_p * w_p)
    total_views = args.shots * (1 + args.aug_count)
    total_tokens = total_views * tokens_per_view
    log(
        f"[2/3] feature_dim={feature_dim}, grid={h_p}x{w_p}, "
        f"views={total_views}, PCA tokens={total_tokens}"
    )

    aug_transform = (
        get_augmentation_transform(AUGMENTATION, IMAGE_RES)
        if args.aug_count > 0
        else None
    )

    def feature_generator():
        for path in selected:
            base = Image.open(path).convert("RGB")
            views = [base]
            if aug_transform is not None:
                for _ in range(args.aug_count):
                    views.append(aug_transform(base))
            for view in views:
                tokens, _, _ = extract_tokens(extractor, view)
                yield tokens.reshape(-1, feature_dim)

    log("[3/3] Fitting official PCA model...")
    pca_model = PCAModel(k=None, ev=PCA_EV, whiten=False)
    pca = pca_model.fit(
        feature_generator,
        feature_dim=feature_dim,
        total_tokens=total_tokens,
        num_batches=total_views,
    )

    config = {
        "method": "SubspaceAD",
        "variant": "paper_giant",
        "source": "CLendering/SubspaceAD official core",
        "model_ckpt": model_ckpt,
        "image_res": IMAGE_RES,
        "layers": LAYERS,
        "aggregation": AGG_METHOD,
        "pca_explained_variance": PCA_EV,
        "augmentation": AUGMENTATION,
        "augmentation_count": int(args.aug_count),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "feature_dim": feature_dim,
        "patch_grid": [h_p, w_p],
        "pca_components": int(pca["k"]),
        "image_score": "mean_top_1_percent",
        "selected_normal_images": [str(p) for p in selected],
    }
    save_pca(model_dir, pca, config)

    log("")
    log("========== FIT DONE ==========")
    log(f"PCA components: {pca['k']}")
    log(f"saved: {model_dir / 'pca_model.npz'}")
    log(f"saved: {model_dir / 'config.json'}")


def save_visuals(image_path: Path, anomaly_map: np.ndarray, output_dir: Path, score: float):
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "anomaly_map.npy"
    np.save(raw_path, anomaly_map.astype(np.float32))

    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    map_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)
    heat_path = output_dir / "heatmap.jpg"
    cv2.imwrite(str(heat_path), heat)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")
    heat_full = cv2.resize(
        heat, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    overlay = cv2.addWeighted(original, 0.55, heat_full, 0.45, 0.0)
    cv2.putText(
        overlay,
        f"SubspaceAD-G score={score:.4f}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.6, min(1.2, min(original.shape[:2]) / 1400.0)),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay_path = output_dir / "overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    return raw_path, heat_path, overlay_path


def inspect_command(args: argparse.Namespace) -> None:
    input_path = resolve_path(args.input)
    images = collect_images(input_path)
    model_dir = resolve_path(args.model_dir) if args.model_dir else default_model_dir(args.product)
    pca, config = load_pca(model_dir)
    model_ckpt = resolve_model_ckpt(
        args.model_ckpt if args.model_ckpt else str(config["model_ckpt"])
    )

    log("========== SubspaceAD Giant INSPECT ==========")
    log(f"input:      {input_path}")
    log(f"images:     {len(images)}")
    log(f"model dir:  {model_dir}")
    log(f"backbone:   {model_ckpt}")
    log("Loading official SubspaceAD FeatureExtractor...")
    extractor = FeatureExtractor(model_ckpt)

    output_root = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_root(args.product)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, image_path in enumerate(images, 1):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()

        pil = Image.open(image_path).convert("RGB")
        tokens, (h_p, w_p), _ = extract_tokens(extractor, pil)
        feature_dim = int(tokens.shape[-1])
        patch_scores = calculate_anomaly_scores(
            tokens.reshape(h_p * w_p, feature_dim),
            pca,
            method="reconstruction",
            drop_k=0,
        )
        anomaly_low = patch_scores.reshape(h_p, w_p)
        anomaly_map = post_process_map(anomaly_low, IMAGE_RES)
        image_score = float(topk_mean(anomaly_map, frac=TOP_FRACTION))

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        image_output = output_root / image_path.stem
        raw_path, heat_path, overlay_path = save_visuals(
            image_path, anomaly_map, image_output, image_score
        )
        result = {
            "image": str(image_path),
            "image_score_top_1pct": image_score,
            "patch_score_min": float(np.min(patch_scores)),
            "patch_score_max": float(np.max(patch_scores)),
            "pca_components": int(pca["k"]),
            "feature_dim": feature_dim,
            "patch_grid": [h_p, w_p],
            "inference_ms": inference_ms,
            "pass_ng": "UNCALIBRATED",
            "raw_map": str(raw_path),
            "heatmap": str(heat_path),
            "overlay": str(overlay_path),
        }
        (image_output / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        rows.append(result)
        log(
            f"[{index}/{len(images)}] {image_path.name} "
            f"score={image_score:.6f} time={inference_ms:.1f} ms"
        )

    if len(rows) > 1:
        (output_root / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    log(f"Output: {output_root}")
    log("PASS/NG remains UNCALIBRATED.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent SubspaceAD DINOv2-Giant fit/inspect workflow."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit and save the Giant PCA model")
    fit.add_argument("--product", required=True)
    fit.add_argument("--normal-dir", required=True)
    fit.add_argument("--shots", type=int, choices=[1, 2, 4], default=4)
    fit.add_argument(
        "--aug-count",
        type=int,
        default=0,
        help="0 for practical first test; use 30 for the paper benchmark setting.",
    )
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--model-dir", default=None)
    fit.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    fit.set_defaults(func=fit_command)

    inspect = sub.add_parser("inspect", help="Inspect one image or every image in a folder")
    inspect.add_argument("--product", required=True)
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--model-dir", default=None)
    inspect.add_argument("--model-ckpt", default=None)
    inspect.add_argument("--output-dir", default=None)
    inspect.set_defaults(func=inspect_command)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "aug_count") and args.aug_count < 0:
        parser.error("--aug-count must be >= 0")
    args.func(args)


if __name__ == "__main__":
    main()
