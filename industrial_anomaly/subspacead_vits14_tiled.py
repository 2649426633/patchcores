from __future__ import annotations

import argparse
import json
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.subspacead.dinov2s_extractor import (
    DINOv2SSubspaceConfig,
    DINOv2SSubspaceExtractor,
)
from app.subspacead.paper_ops import (
    PCAModel,
    calculate_anomaly_scores,
    min_max_norm,
    post_process_map,
    topk_mean,
)
from app.subspacead.tiling import compute_tile_windows


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PCA_EV = 0.99
DEFAULT_IMAGE_SIZE = 448
DEFAULT_BLOCKS = (7, 8)
TOP_FRACTION = 0.01
DEFAULT_TILE_FRACTION = 0.50
DEFAULT_TILE_OVERLAP = 0.25
DEFAULT_BATCH_SIZE = 4


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
    return HERE / "products" / product / "models" / "subspacead_vits14_tiled"


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


def _angles(seed: int, count: int) -> list[float | None]:
    rng = random.Random(seed)
    values: list[float | None] = [None]
    values.extend(rng.uniform(0.0, 345.0) for _ in range(count))
    return values


def _tile_batch(image: Image.Image, tile_fraction: float, overlap: float):
    windows = compute_tile_windows(
        image.size,
        tile_fraction=float(tile_fraction),
        overlap=float(overlap),
    )
    tiles = [image.crop((w.x1, w.y1, w.x2, w.y2)) for w in windows]
    return windows, tiles


def _iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def fit_command(args: argparse.Namespace) -> None:
    normal_dir = resolve_path(args.normal_dir)
    model_dir = (
        resolve_path(args.model_dir)
        if args.model_dir
        else default_model_dir(args.product)
    )
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

    print("========== SubspaceAD DINOv2-S TILED FIT ==========")
    print(f"product:        {args.product}")
    print(f"normal dir:     {normal_dir}")
    print(f"shots:          {args.shots}")
    print(f"augmentations:  {args.aug_count} rotations / image")
    print(f"resolution:     {DEFAULT_IMAGE_SIZE}")
    print(f"blocks:         {DEFAULT_BLOCKS}")
    print(f"tile fraction:  {args.tile_fraction}")
    print(f"tile overlap:   {args.tile_overlap}")
    print(f"batch size:     {args.batch_size}")
    print(f"PCA EV:         {PCA_EV}")
    print(f"model dir:      {model_dir}")
    for i, path in enumerate(selected, 1):
        print(f"normal[{i}]:     {path.name}")

    extractor = make_extractor(
        device=args.device,
        repo_dir=args.dinov2_repo,
        weights_path=args.weights,
    )

    probe_image = Image.open(selected[0]).convert("RGB")
    _, probe_tiles = _tile_batch(
        probe_image, args.tile_fraction, args.tile_overlap
    )
    probe_tokens = extractor.extract_tokens([probe_tiles[0]])
    _, h_p, w_p, feature_dim = probe_tokens.shape
    tokens_per_tile = h_p * w_p

    angle_plan: dict[str, list[float | None]] = {}
    total_tiles_per_pass = 0
    for image_index, path in enumerate(selected):
        plan = _angles(args.seed + image_index * 1009, args.aug_count)
        angle_plan[str(path)] = plan
        with Image.open(path) as im:
            windows = compute_tile_windows(
                im.size,
                tile_fraction=args.tile_fraction,
                overlap=args.tile_overlap,
            )
        total_tiles_per_pass += len(windows) * len(plan)

    total_tokens = total_tiles_per_pass * tokens_per_tile
    num_batches = 0
    for path in selected:
        with Image.open(path) as im:
            n_tiles = len(
                compute_tile_windows(
                    im.size,
                    tile_fraction=args.tile_fraction,
                    overlap=args.tile_overlap,
                )
            )
        num_batches += (
            (n_tiles + args.batch_size - 1) // args.batch_size
        ) * len(angle_plan[str(path)])

    print(
        f"feature_dim={feature_dim}, grid={h_p}x{w_p}, "
        f"tiles/pass={total_tiles_per_pass}, PCA tokens={total_tokens}"
    )

    def feature_generator():
        for path in selected:
            base = Image.open(path).convert("RGB")
            for angle in angle_plan[str(path)]:
                view = base if angle is None else TF.rotate(base, angle=float(angle))
                _, tiles = _tile_batch(view, args.tile_fraction, args.tile_overlap)
                for batch in _iter_batches(tiles, args.batch_size):
                    tokens = extractor.extract_tokens(batch)
                    yield tokens.reshape(-1, feature_dim)

    print("Fitting tiled PCA (two streaming passes, no sklearn)...")
    pca_model = PCAModel(
        k=None,
        ev=PCA_EV,
        whiten=False,
        device=str(extractor.device),
    )
    pca = pca_model.fit(
        feature_generator,
        feature_dim=feature_dim,
        total_tokens=total_tokens,
        num_batches=num_batches,
    )

    config = {
        "method": "SubspaceAD",
        "variant": "industrial_dinov2_vits14_tiled",
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
        "augmentation_count": int(args.aug_count),
        "shots": int(args.shots),
        "seed": int(args.seed),
        "image_score": "mean_top_1_percent",
        "gaussian_sigma": 4.0,
        "inspection_mode": "tiled",
        "tile_fraction": float(args.tile_fraction),
        "tile_overlap": float(args.tile_overlap),
        "tile_merge": "weighted_average",
        "selected_normal_images": [str(p) for p in selected],
        "runtime": "sklearn_free_industrial_adapter",
        "note": (
            "Industrial high-resolution extension. The vendored SubspaceAD source remains unchanged; "
            "the adapter implements the same standard PCA reconstruction path without importing "
            "scikit-learn, avoiding mixed OpenMP runtimes on Windows."
        ),
    }
    save_pca(model_dir, pca, config)

    print("========== TILED FIT DONE ==========")
    print(f"PCA components: {pca['k']}")
    print(f"saved: {model_dir / 'pca_model.npz'}")
    print(f"saved: {model_dir / 'config.json'}")


def _blend_weight(height: int, width: int) -> np.ndarray:
    if height < 3 or width < 3:
        return np.ones((height, width), dtype=np.float32)
    wy = np.hanning(height).astype(np.float32)
    wx = np.hanning(width).astype(np.float32)
    weight = np.outer(wy, wx)
    return np.maximum(weight, 0.05).astype(np.float32)


def _bbox_from_full_map(
    anomaly_map: np.ndarray,
    relative_threshold: float,
    min_area: int,
) -> tuple[int, int, int, int] | None:
    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    mask = (normalized >= float(relative_threshold)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return None
    candidates = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            candidates.append((area, label))
    if not candidates:
        return None
    _, best = max(candidates)
    x = int(stats[best, cv2.CC_STAT_LEFT])
    y = int(stats[best, cv2.CC_STAT_TOP])
    w = int(stats[best, cv2.CC_STAT_WIDTH])
    h = int(stats[best, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def _save_visuals(
    image_path: Path,
    full_map: np.ndarray,
    output_dir: Path,
    bbox: tuple[int, int, int, int] | None,
    image_score: float,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "anomaly_map.npy"
    np.save(raw_path, full_map.astype(np.float32))

    normalized = np.asarray(min_max_norm(full_map), dtype=np.float32)
    heat_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_path = output_dir / "heatmap.jpg"
    cv2.imwrite(str(heat_path), heat)

    original = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"OpenCV failed to read: {image_path}")
    overlay = cv2.addWeighted(original, 0.60, heat, 0.40, 0.0)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        thickness = max(2, int(round(min(original.shape[:2]) / 400.0)))
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), thickness)
    cv2.putText(
        overlay,
        f"SubspaceAD-S tiled score={image_score:.4f}",
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
    model_dir = (
        resolve_path(args.model_dir)
        if args.model_dir
        else default_model_dir(args.product)
    )
    pca, config = load_pca(model_dir)

    if str(config.get("inspection_mode", "")) != "tiled":
        raise RuntimeError(
            "This command requires a tiled PCA model. Run the tiled 'fit' command first."
        )

    repo_dir = args.dinov2_repo or str(
        config.get("dinov2_repo", "third_party/dinov2")
    )
    weights = args.weights or str(
        config.get("source_weights", "weights/dinov2_vits14_pretrain.pth")
    )
    image_size = int(config.get("image_size", DEFAULT_IMAGE_SIZE))
    blocks = tuple(
        int(v) for v in config.get("local_block_indices", DEFAULT_BLOCKS)
    )
    tile_fraction = (
        float(args.tile_fraction)
        if args.tile_fraction is not None
        else float(config.get("tile_fraction", DEFAULT_TILE_FRACTION))
    )
    tile_overlap = (
        float(args.tile_overlap)
        if args.tile_overlap is not None
        else float(config.get("tile_overlap", DEFAULT_TILE_OVERLAP))
    )

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
        else HERE
        / "outputs"
        / args.product
        / "subspacead_vits14_tiled"
        / input_path.stem
    )
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for image_index, image_path in enumerate(images, 1):
        pil = Image.open(image_path).convert("RGB")
        width, height = pil.size
        windows, tiles = _tile_batch(pil, tile_fraction, tile_overlap)
        merged_sum = np.zeros((height, width), dtype=np.float32)
        merged_weight = np.zeros((height, width), dtype=np.float32)
        tile_records = []

        if extractor.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()

        tile_cursor = 0
        for batch in _iter_batches(tiles, args.batch_size):
            token_batch = extractor.extract_tokens(batch)
            for local_index in range(token_batch.shape[0]):
                window = windows[tile_cursor]
                tokens = token_batch[local_index]
                h_p, w_p, feature_dim = tokens.shape
                scores = calculate_anomaly_scores(
                    tokens.reshape(h_p * w_p, feature_dim),
                    pca,
                    method="reconstruction",
                    drop_k=0,
                )
                tile_low = scores.reshape(h_p, w_p)
                tile_map_448 = post_process_map(tile_low, image_size)
                tile_h = window.y2 - window.y1
                tile_w = window.x2 - window.x1
                tile_map = cv2.resize(
                    tile_map_448.astype(np.float32),
                    (tile_w, tile_h),
                    interpolation=cv2.INTER_LINEAR,
                )
                weight = _blend_weight(tile_h, tile_w)
                merged_sum[
                    window.y1 : window.y2, window.x1 : window.x2
                ] += tile_map * weight
                merged_weight[
                    window.y1 : window.y2, window.x1 : window.x2
                ] += weight
                tile_records.append(
                    {
                        "tile_index": tile_cursor,
                        "bbox": [window.x1, window.y1, window.x2, window.y2],
                        "score_top_1pct": float(
                            topk_mean(tile_map_448, frac=TOP_FRACTION)
                        ),
                        "patch_score_max": float(np.max(scores)),
                    }
                )
                tile_cursor += 1

        full_map = merged_sum / np.maximum(merged_weight, 1e-6)
        image_score = float(topk_mean(full_map, frac=TOP_FRACTION))

        if extractor.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        bbox = _bbox_from_full_map(
            full_map,
            args.bbox_relative_threshold,
            args.min_region_area,
        )
        image_output = (
            output_root if len(images) == 1 else output_root / image_path.stem
        )
        raw_path, heat_path, overlay_path = _save_visuals(
            image_path,
            full_map,
            image_output,
            bbox,
            image_score,
        )
        (image_output / "tiles.json").write_text(
            json.dumps(tile_records, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        result = {
            "image": str(image_path),
            "method": "SubspaceAD",
            "variant": "industrial_dinov2_vits14_tiled",
            "image_score_top_1pct": image_score,
            "bbox_relative_threshold": float(args.bbox_relative_threshold),
            "min_region_area": int(args.min_region_area),
            "bbox_original_image": list(bbox) if bbox is not None else None,
            "tile_fraction": tile_fraction,
            "tile_overlap": tile_overlap,
            "tile_count": len(tiles),
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
            f"[{image_index}/{len(images)}] {image_path.name} "
            f"tiles={len(tiles)} score={image_score:.6f} "
            f"bbox={result['bbox_original_image']} time={inference_ms:.1f} ms"
        )

    if len(rows) > 1:
        (output_root / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(f"Output: {output_root.resolve()}")
    print("PASS/NG remains UNCALIBRATED; bbox threshold is localization-only.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="High-resolution tiled industrial SubspaceAD with local DINOv2-S/14."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit PCA from tiled normal images")
    fit.add_argument("--product", required=True)
    fit.add_argument("--normal-dir", required=True)
    fit.add_argument("--shots", type=int, choices=[1, 2, 4], default=4)
    fit.add_argument("--aug-count", type=int, default=0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--device", default=None)
    fit.add_argument("--model-dir", default=None)
    fit.add_argument("--dinov2-repo", default="third_party/dinov2")
    fit.add_argument("--weights", default="weights/dinov2_vits14_pretrain.pth")
    fit.add_argument("--tile-fraction", type=float, default=DEFAULT_TILE_FRACTION)
    fit.add_argument("--tile-overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    fit.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    fit.set_defaults(func=fit_command)

    inspect = sub.add_parser(
        "inspect", help="Inspect one high-resolution image or folder"
    )
    inspect.add_argument("--product", required=True)
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--device", default=None)
    inspect.add_argument("--model-dir", default=None)
    inspect.add_argument("--output-dir", default=None)
    inspect.add_argument("--dinov2-repo", default=None)
    inspect.add_argument("--weights", default=None)
    inspect.add_argument("--tile-fraction", type=float, default=None)
    inspect.add_argument("--tile-overlap", type=float, default=None)
    inspect.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    inspect.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    inspect.add_argument("--min-region-area", type=int, default=64)
    inspect.set_defaults(func=inspect_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "aug_count") and args.aug_count < 0:
        parser.error("--aug-count must be >= 0")
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if (
        hasattr(args, "tile_fraction")
        and args.tile_fraction is not None
        and not (0.2 <= args.tile_fraction <= 1.0)
    ):
        parser.error("--tile-fraction must be in [0.2,1.0]")
    if (
        hasattr(args, "tile_overlap")
        and args.tile_overlap is not None
        and not (0.0 <= args.tile_overlap < 0.9)
    ):
        parser.error("--tile-overlap must be in [0,0.9)")
    if hasattr(args, "bbox_relative_threshold") and not (
        0.0 < args.bbox_relative_threshold <= 1.0
    ):
        parser.error("--bbox-relative-threshold must be in (0,1]")
    args.func(args)


if __name__ == "__main__":
    main()
