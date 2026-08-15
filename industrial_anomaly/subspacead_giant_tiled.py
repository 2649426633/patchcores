from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.subspacead.giant_extractor import (
    DINOv2GiantIndustrialConfig,
    DINOv2GiantIndustrialExtractor,
)
from app.subspacead.paper_ops import (
    PCAModel,
    calculate_anomaly_scores,
    min_max_norm,
    post_process_map,
    topk_mean,
)
from app.subspacead.tiling import TileWindow, compute_tile_windows


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
IMAGE_RES = 672
LAYERS = (-12, -13, -14, -15, -16, -17, -18)
PCA_EV = 0.99
TOP_FRACTION = 0.01
DEFAULT_MODEL_CKPT = "weights/subspacead/dinov2-with-registers-giant"
DEFAULT_TILE_FRACTION = 0.75
DEFAULT_TILE_OVERLAP = 0.25
DEFAULT_BATCH_SIZE = 1


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
    return HERE / "products" / product / "models" / "subspacead_giant_tiled"


def default_output_root(product: str) -> Path:
    return HERE / "outputs" / product / "subspacead_giant_tiled"


def parse_shots(value: str, available: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return available
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError("--shots must be a positive integer or 'all'") from exc
    if count <= 0:
        raise ValueError("--shots must be > 0")
    if count > available:
        raise ValueError(f"Requested {count} normals but only {available} exist")
    return count


def make_extractor(model_ckpt: str, device: str | None):
    cfg = DINOv2GiantIndustrialConfig(
        model_ckpt=model_ckpt,
        image_size=IMAGE_RES,
        layers=LAYERS,
        aggregation="mean",
    )
    extractor = DINOv2GiantIndustrialExtractor(device=device, config=cfg)
    extractor.load()
    return extractor


def _iter_batches(items: list, batch_size: int):
    for start in range(0, len(items), batch_size):
        yield start, items[start : start + batch_size]


def _save_pca(path: Path, pca: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        mu=np.asarray(pca["mu"], dtype=np.float64),
        components=np.asarray(pca["components"], dtype=np.float64),
        eigvals=np.asarray(pca["eigvals"], dtype=np.float64),
        sqrt_eig=np.asarray(pca["sqrt_eig"], dtype=np.float64),
        k=np.asarray([int(pca["k"])], dtype=np.int64),
        eps=np.asarray([float(pca["eps"])], dtype=np.float64),
    )


def _load_pca(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    eigvals = np.asarray(data["eigvals"], dtype=np.float64)
    eps = float(np.asarray(data["eps"]).reshape(-1)[0])
    return {
        "mu": np.asarray(data["mu"], dtype=np.float64),
        "components": np.asarray(data["components"], dtype=np.float64),
        "eigvals": eigvals,
        "sqrt_eig": np.asarray(data["sqrt_eig"], dtype=np.float64),
        "k": int(np.asarray(data["k"]).reshape(-1)[0]),
        "eps": eps,
        "whiten": False,
        "cov_Z_inv": np.diag(1.0 / (eigvals + eps)),
    }


def _window_to_list(window: TileWindow) -> list[int]:
    return [window.x1, window.y1, window.x2, window.y2]


def _list_to_window(values: list[int]) -> TileWindow:
    return TileWindow(*(int(v) for v in values))


def _validate_same_size(paths: list[Path], expected: tuple[int, int] | None = None) -> tuple[int, int]:
    reference = expected
    for path in paths:
        with Image.open(path) as image:
            size = image.size
        if reference is None:
            reference = size
        elif size != reference:
            raise RuntimeError(
                "Spatial tiled PCA requires a fixed camera/image size. "
                f"Expected {reference}, got {size} for {path.name}."
            )
    if reference is None:
        raise RuntimeError("No images")
    return reference


def _blend_weight(height: int, width: int) -> np.ndarray:
    if height < 3 or width < 3:
        return np.ones((height, width), dtype=np.float32)
    wy = np.hanning(height).astype(np.float32)
    wx = np.hanning(width).astype(np.float32)
    weight = np.outer(wy, wx)
    return np.maximum(weight, 0.05).astype(np.float32)


def _bbox_from_map(
    anomaly_map: np.ndarray,
    relative_threshold: float,
    min_area: int,
) -> tuple[int, int, int, int] | None:
    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    mask = (normalized >= float(relative_threshold)).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            candidates.append((area, label))
    if not candidates:
        return None
    _, label = max(candidates)
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def fit_command(args: argparse.Namespace) -> None:
    normal_dir = resolve_path(args.normal_dir)
    model_dir = resolve_path(args.model_dir) if args.model_dir else default_model_dir(args.product)
    all_normals = collect_images(normal_dir)
    shot_count = parse_shots(args.shots, len(all_normals))

    random.seed(args.seed)
    selected = all_normals.copy()
    random.shuffle(selected)
    selected = selected[:shot_count]
    reference_size = _validate_same_size(selected)
    windows = compute_tile_windows(
        reference_size,
        tile_fraction=args.tile_fraction,
        overlap=args.tile_overlap,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    cache_root = model_dir / "_feature_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    log("========== SubspaceAD Giant SPATIAL-TILED FIT ==========")
    log(f"product:        {args.product}")
    log(f"normal dir:     {normal_dir}")
    log(f"normal images:  {shot_count}/{len(all_normals)}")
    log(f"image size:     {reference_size[0]}x{reference_size[1]}")
    log(f"tile fraction:  {args.tile_fraction}")
    log(f"tile overlap:   {args.tile_overlap}")
    log(f"tile count:     {len(windows)}")
    log(f"tile input:     {IMAGE_RES}x{IMAGE_RES}")
    log(f"layers:         {list(LAYERS)}")
    log(f"PCA EV:         {PCA_EV}")
    log(f"batch size:     {args.batch_size}")
    log("mode:           fixed-position PCA per tile")
    log("cache:          extract Giant features once; PCA reuses cached features")

    extractor = make_extractor(args.model_ckpt, args.device)
    tile_models = []
    feature_dim = None
    patch_grid = None

    for tile_index, window in enumerate(windows):
        log("")
        log(f"--- tile {tile_index + 1}/{len(windows)} bbox={_window_to_list(window)} ---")
        tile_cache = cache_root / f"tile_{tile_index:03d}"
        tile_cache.mkdir(parents=True, exist_ok=True)
        cache_files: list[Path] = []

        pending: list[tuple[int, Path]] = []
        for sample_index, image_path in enumerate(selected):
            cache_file = tile_cache / f"sample_{sample_index:04d}.npy"
            cache_files.append(cache_file)
            if args.rebuild_cache or not cache_file.exists():
                pending.append((sample_index, image_path))

        if pending:
            log(f"Extracting {len(pending)} uncached Giant tile features...")
            for start in range(0, len(pending), args.batch_size):
                batch_meta = pending[start : start + args.batch_size]
                batch_tiles = []
                for _, image_path in batch_meta:
                    with Image.open(image_path) as image:
                        rgb = image.convert("RGB")
                        batch_tiles.append(
                            rgb.crop((window.x1, window.y1, window.x2, window.y2))
                        )
                tokens_batch = extractor.extract_tokens(batch_tiles)
                for local_index, (sample_index, _) in enumerate(batch_meta):
                    tokens = tokens_batch[local_index]
                    h_p, w_p, dim = tokens.shape
                    if feature_dim is None:
                        feature_dim = int(dim)
                        patch_grid = [int(h_p), int(w_p)]
                    np.save(
                        tile_cache / f"sample_{sample_index:04d}.npy",
                        tokens.reshape(-1, dim).astype(np.float32),
                    )
                log(
                    f"  cached {min(start + len(batch_meta), len(pending))}/{len(pending)}"
                )
        else:
            log("Using existing feature cache.")

        if feature_dim is None:
            probe = np.load(cache_files[0], mmap_mode="r")
            feature_dim = int(probe.shape[1])
            token_count = int(probe.shape[0])
            side = int(round(token_count ** 0.5))
            patch_grid = [side, side]

        token_count_per_sample = int(np.load(cache_files[0], mmap_mode="r").shape[0])
        total_tokens = token_count_per_sample * len(cache_files)

        def feature_generator():
            for cache_file in cache_files:
                yield np.asarray(np.load(cache_file, mmap_mode="r"), dtype=np.float32)

        log(
            f"Fitting tile PCA: samples={len(cache_files)}, "
            f"tokens={total_tokens}, dim={feature_dim}"
        )
        pca_model = PCAModel(k=None, ev=PCA_EV, whiten=False, device=args.device)
        pca = pca_model.fit(
            feature_generator,
            feature_dim=int(feature_dim),
            total_tokens=total_tokens,
            num_batches=len(cache_files),
        )
        pca_name = f"tile_{tile_index:03d}.npz"
        _save_pca(model_dir / pca_name, pca)
        tile_models.append(
            {
                "tile_index": tile_index,
                "bbox": _window_to_list(window),
                "pca_file": pca_name,
                "pca_components": int(pca["k"]),
            }
        )
        log(f"tile {tile_index}: PCA k={pca['k']} saved={pca_name}")

        del pca_model, pca
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not args.keep_cache:
            shutil.rmtree(tile_cache, ignore_errors=True)

    if not args.keep_cache:
        shutil.rmtree(cache_root, ignore_errors=True)

    config = {
        "method": "SubspaceAD",
        "variant": "industrial_giant_spatial_tiled",
        "backbone": "dinov2-with-registers-giant",
        "model_ckpt": args.model_ckpt,
        "image_res": IMAGE_RES,
        "layers": list(LAYERS),
        "aggregation": "mean",
        "attention_output": False,
        "pca_explained_variance": PCA_EV,
        "pca_scope": "independent_per_fixed_tile_position",
        "reference_image_size": [reference_size[0], reference_size[1]],
        "tile_fraction": float(args.tile_fraction),
        "tile_overlap": float(args.tile_overlap),
        "tile_merge": "hann_weighted_average",
        "tile_models": tile_models,
        "feature_dim": int(feature_dim),
        "patch_grid": patch_grid,
        "normal_count": shot_count,
        "seed": int(args.seed),
        "selected_normal_images": [str(path) for path in selected],
        "image_score": "mean_top_1_percent_of_merged_map",
        "note": (
            "Industrial adaptation for a fixed camera. It preserves the DINOv2-Giant "
            "backbone/layers and reconstruction-residual PCA idea, but is not the paper's "
            "strict full-image 4-shot benchmark protocol. third_party/subspacead is unchanged."
        ),
    }
    (model_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log("")
    log("========== GIANT SPATIAL-TILED FIT DONE ==========")
    log(f"tiles:  {len(tile_models)}")
    log(f"model:  {model_dir}")
    log("==================================================")


def _save_visuals(
    image_path: Path,
    full_map: np.ndarray,
    output_dir: Path,
    score: float,
    bbox: tuple[int, int, int, int] | None,
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
        f"SubspaceAD-G spatial-tiled score={score:.4f}",
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
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Run fit first: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    expected_size = tuple(int(v) for v in config["reference_image_size"])
    _validate_same_size(images, expected=expected_size)
    tile_models = config["tile_models"]
    windows = [_list_to_window(item["bbox"]) for item in tile_models]
    pcas = [_load_pca(model_dir / item["pca_file"]) for item in tile_models]

    model_ckpt = args.model_ckpt or str(config.get("model_ckpt", DEFAULT_MODEL_CKPT))
    extractor = make_extractor(model_ckpt, args.device)
    output_root = (
        resolve_path(args.output_dir)
        if args.output_dir
        else default_output_root(args.product)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    log("========== SubspaceAD Giant SPATIAL-TILED INSPECT ==========")
    log(f"input:       {input_path}")
    log(f"images:      {len(images)}")
    log(f"tiles/image: {len(windows)}")
    log(f"model dir:   {model_dir}")

    rows = []
    for image_index, image_path in enumerate(images, 1):
        with Image.open(image_path) as image:
            pil = image.convert("RGB")
            width, height = pil.size
            tiles = [
                pil.crop((w.x1, w.y1, w.x2, w.y2))
                for w in windows
            ]

        merged_sum = np.zeros((height, width), dtype=np.float32)
        merged_weight = np.zeros((height, width), dtype=np.float32)
        tile_records = []

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        started = time.perf_counter()

        for batch_start, batch_tiles in _iter_batches(tiles, args.batch_size):
            token_batch = extractor.extract_tokens(batch_tiles)
            for local_index in range(token_batch.shape[0]):
                tile_index = batch_start + local_index
                window = windows[tile_index]
                tokens = token_batch[local_index]
                h_p, w_p, feature_dim = tokens.shape
                patch_scores = calculate_anomaly_scores(
                    tokens.reshape(h_p * w_p, feature_dim),
                    pcas[tile_index],
                    method="reconstruction",
                    drop_k=0,
                )
                low_map = patch_scores.reshape(h_p, w_p)
                map_672 = post_process_map(low_map, IMAGE_RES)
                tile_map = cv2.resize(
                    map_672.astype(np.float32),
                    (window.width, window.height),
                    interpolation=cv2.INTER_LINEAR,
                )
                weight = _blend_weight(window.height, window.width)
                merged_sum[window.y1 : window.y2, window.x1 : window.x2] += tile_map * weight
                merged_weight[window.y1 : window.y2, window.x1 : window.x2] += weight
                tile_records.append(
                    {
                        "tile_index": tile_index,
                        "bbox": _window_to_list(window),
                        "score_top_1pct": float(topk_mean(map_672, frac=TOP_FRACTION)),
                        "patch_score_max": float(np.max(patch_scores)),
                        "pca_components": int(pcas[tile_index]["k"]),
                    }
                )

        full_map = merged_sum / np.maximum(merged_weight, 1e-6)
        image_score = float(topk_mean(full_map, frac=TOP_FRACTION))
        bbox = _bbox_from_map(
            full_map,
            relative_threshold=args.bbox_relative_threshold,
            min_area=args.min_region_area,
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        image_output = output_root / image_path.stem
        raw_path, heat_path, overlay_path = _save_visuals(
            image_path, full_map, image_output, image_score, bbox
        )
        tile_records.sort(key=lambda item: item["score_top_1pct"], reverse=True)
        (image_output / "tiles.json").write_text(
            json.dumps(tile_records, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        result = {
            "image": str(image_path),
            "method": "SubspaceAD",
            "variant": "industrial_giant_spatial_tiled",
            "image_score_top_1pct": image_score,
            "bbox_original_image": list(bbox) if bbox is not None else None,
            "bbox_relative_threshold": float(args.bbox_relative_threshold),
            "min_region_area": int(args.min_region_area),
            "tile_count": len(windows),
            "top_tile": tile_records[0] if tile_records else None,
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
            f"[{image_index}/{len(images)}] {image_path.name} "
            f"score={image_score:.6f} bbox={result['bbox_original_image']} "
            f"top_tile={result['top_tile']['tile_index'] if result['top_tile'] else None} "
            f"time={inference_ms:.1f} ms"
        )

    if len(rows) > 1:
        (output_root / "results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    log(f"Output: {output_root}")
    log("PASS/NG remains UNCALIBRATED; first verify normal/abnormal separation and heatmap location.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Industrial high-resolution SubspaceAD: DINOv2-Giant + fixed-position "
            "overlapping tiles + one PCA per tile position."
        )
    )
    sub = parser.add_subparsers(dest="command", required=True)

    fit = sub.add_parser("fit", help="Fit independent PCA models for fixed tile positions")
    fit.add_argument("--product", required=True)
    fit.add_argument("--normal-dir", required=True)
    fit.add_argument(
        "--shots",
        default="8",
        help="Number of normal images, or 'all'. Unlike the paper benchmark this industrial mode is not limited to 1/2/4.",
    )
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--model-dir", default=None)
    fit.add_argument("--model-ckpt", default=DEFAULT_MODEL_CKPT)
    fit.add_argument("--device", default=None)
    fit.add_argument("--tile-fraction", type=float, default=DEFAULT_TILE_FRACTION)
    fit.add_argument("--tile-overlap", type=float, default=DEFAULT_TILE_OVERLAP)
    fit.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    fit.add_argument("--keep-cache", action="store_true")
    fit.add_argument("--rebuild-cache", action="store_true")
    fit.set_defaults(func=fit_command)

    inspect = sub.add_parser("inspect", help="Inspect one image or a folder")
    inspect.add_argument("--product", required=True)
    inspect.add_argument("--input", required=True)
    inspect.add_argument("--model-dir", default=None)
    inspect.add_argument("--model-ckpt", default=None)
    inspect.add_argument("--device", default=None)
    inspect.add_argument("--output-dir", default=None)
    inspect.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    inspect.add_argument("--bbox-relative-threshold", type=float, default=0.85)
    inspect.add_argument("--min-region-area", type=int, default=64)
    inspect.set_defaults(func=inspect_command)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        parser.error("--batch-size must be > 0")
    if hasattr(args, "tile_fraction") and not (0.2 <= args.tile_fraction <= 1.0):
        parser.error("--tile-fraction must be in [0.2, 1.0]")
    if hasattr(args, "tile_overlap") and not (0.0 <= args.tile_overlap < 0.9):
        parser.error("--tile-overlap must be in [0.0, 0.9)")
    if hasattr(args, "bbox_relative_threshold") and not (
        0.0 < args.bbox_relative_threshold <= 1.0
    ):
        parser.error("--bbox-relative-threshold must be in (0,1]")
    args.func(args)


if __name__ == "__main__":
    main()
