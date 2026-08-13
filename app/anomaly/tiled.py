from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .image_dataset import SUPPORTED_EXTENSIONS
from .postprocessing import extract_regions_from_map, normalize_anomaly_map


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
INSPECTION_CONFIG_FILENAME = "inspection_config.json"


@dataclass(frozen=True)
class TileWindow:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _axis_starts(length: int, tile_size: int, overlap: float) -> list[int]:
    if tile_size >= length:
        return [0]
    stride = max(1.0, tile_size * (1.0 - overlap))
    count = max(2, int(math.ceil((length - tile_size) / stride)) + 1)
    values = np.linspace(0, length - tile_size, count)
    starts = sorted({int(round(v)) for v in values})
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def compute_tile_windows(
    image_size: tuple[int, int],
    tile_fraction: float = 0.75,
    overlap: float = 0.25,
) -> list[TileWindow]:
    """Cover the complete image with overlapping square tiles.

    ``tile_fraction`` is relative to the shorter image side.  For the user's
    5472x3648 phone images, 0.75 gives 2736px square tiles and six overlapping
    views, preserving much more local detail than resizing the full image to 320.
    """
    w, h = image_size
    if not (0.2 <= tile_fraction <= 1.0):
        raise ValueError("tile_fraction must be in [0.2, 1.0]")
    if not (0.0 <= overlap < 0.9):
        raise ValueError("overlap must be in [0.0, 0.9)")

    tile_size = max(32, int(round(min(w, h) * tile_fraction)))
    tile_size = min(tile_size, w, h)
    xs = _axis_starts(w, tile_size, overlap)
    ys = _axis_starts(h, tile_size, overlap)
    return [
        TileWindow(x, y, x + tile_size, y + tile_size)
        for y in ys
        for x in xs
    ]


def build_direct_square_transform(imagesize: int):
    return transforms.Compose(
        [
            transforms.Resize((imagesize, imagesize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


class TiledNormalImageDataset(Dataset):
    """Normal-only dataset that expands each source image into overlapping tiles."""

    def __init__(
        self,
        image_dir: str | Path,
        imagesize: int = 320,
        tile_fraction: float = 0.75,
        overlap: float = 0.25,
    ):
        self.image_dir = Path(image_dir)
        if not self.image_dir.is_dir():
            raise NotADirectoryError(self.image_dir)

        self.image_paths = sorted(
            p
            for p in self.image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found in: {self.image_dir}")

        self.imagesize = int(imagesize)
        self.tile_fraction = float(tile_fraction)
        self.overlap = float(overlap)
        self.transform = build_direct_square_transform(self.imagesize)

        samples: list[tuple[Path, TileWindow]] = []
        for path in self.image_paths:
            with Image.open(path) as im:
                windows = compute_tile_windows(
                    im.size,
                    tile_fraction=self.tile_fraction,
                    overlap=self.overlap,
                )
            samples.extend((path, window) for window in windows)
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, window = self.samples[index]
        image = Image.open(path).convert("RGB")
        tile = image.crop((window.x1, window.y1, window.x2, window.y2))
        return {
            "image": self.transform(tile),
            "image_path": str(path),
            "tile_bbox": torch.tensor(
                [window.x1, window.y1, window.x2, window.y2], dtype=torch.int32
            ),
        }


def create_tiled_normal_dataloader(
    image_dir: str | Path,
    batch_size: int = 8,
    num_workers: int = 0,
    imagesize: int = 320,
    tile_fraction: float = 0.75,
    overlap: float = 0.25,
):
    dataset = TiledNormalImageDataset(
        image_dir=image_dir,
        imagesize=imagesize,
        tile_fraction=tile_fraction,
        overlap=overlap,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )


def save_inspection_config(
    model_dir: str | Path,
    *,
    mode: str,
    imagesize: int,
    tile_fraction: float | None = None,
    tile_overlap: float | None = None,
) -> Path:
    model_dir = Path(model_dir)
    payload = {
        "format_version": 1,
        "mode": mode,
        "imagesize": int(imagesize),
    }
    if tile_fraction is not None:
        payload["tile_fraction"] = float(tile_fraction)
    if tile_overlap is not None:
        payload["tile_overlap"] = float(tile_overlap)
    path = model_dir / INSPECTION_CONFIG_FILENAME
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_inspection_config(model_dir: str | Path) -> dict:
    path = Path(model_dir) / INSPECTION_CONFIG_FILENAME
    if not path.exists():
        return {"format_version": 0, "mode": "center_crop"}
    return json.loads(path.read_text(encoding="utf-8"))


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return float(inter / max(1.0, area_a + area_b - inter))


def _union_bbox(a: tuple[int, int, int, int], b: tuple[int, int, int, int]):
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def _merge_overlapping_regions(regions: list[dict], iou_threshold: float = 0.15) -> list[dict]:
    merged: list[dict] = []
    for region in sorted(regions, key=lambda r: r["rank_score"], reverse=True):
        matched = None
        for existing in merged:
            if _bbox_iou(region["bbox"], existing["bbox"]) >= iou_threshold:
                matched = existing
                break
        if matched is None:
            merged.append(dict(region))
        else:
            matched["bbox"] = _union_bbox(matched["bbox"], region["bbox"])
            matched["rank_score"] = max(matched["rank_score"], region["rank_score"])
            matched["tile_score"] = max(matched["tile_score"], region["tile_score"])
            matched["evidence"] = max(matched["evidence"], region["evidence"])
            matched["peak"] = max(matched["peak"], region["peak"])
            matched["merged_detections"] = int(matched.get("merged_detections", 1)) + 1
    return sorted(merged, key=lambda r: r["rank_score"], reverse=True)


def crop_square_with_margin(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    margin: float = 0.50,
) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = bbox
    bw, bh = max(1, x2 - x1), max(1, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(8, int(math.ceil(max(bw, bh) * (1.0 + 2.0 * margin))))

    left = int(math.floor(cx - side / 2.0))
    top = int(math.floor(cy - side / 2.0))
    right, bottom = left + side, top + side

    arr = np.asarray(image, dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
    fill = tuple(int(v) for v in np.median(border.astype(np.float32), axis=0))
    out = Image.new("RGB", (side, side), fill)

    sl, st = max(0, left), max(0, top)
    sr, sb = min(w, right), min(h, bottom)
    if sr > sl and sb > st:
        out.paste(image.crop((sl, st, sr, sb)), (sl - left, st - top))
    return out


def inspect_tiled_patchcore(
    patchcore_adapter,
    image_path: str | Path,
    *,
    tile_fraction: float = 0.75,
    overlap: float = 0.25,
    relative_threshold: float = 0.78,
    min_area: int = 8,
    max_regions_per_tile: int = 4,
    max_regions: int = 8,
    min_global_ratio: float = 0.55,
    merge_iou: float = 0.15,
) -> dict:
    """Run PatchCore over overlapping full-image tiles and merge anomaly regions."""
    image_path = Path(image_path)
    original = Image.open(image_path).convert("RGB")
    windows = compute_tile_windows(
        original.size,
        tile_fraction=tile_fraction,
        overlap=overlap,
    )
    transform = build_direct_square_transform(patchcore_adapter.config.imagesize)

    all_regions: list[dict] = []
    tile_results: list[dict] = []
    best_tile_score = float("-inf")

    for tile_index, window in enumerate(windows):
        tile = original.crop((window.x1, window.y1, window.x2, window.y2))
        tensor = transform(tile).unsqueeze(0)
        result = patchcore_adapter.predict(tensor)
        anomaly_map = np.asarray(result["anomaly_map"], dtype=np.float32)
        tile_score = float(result["anomaly_score"])
        best_tile_score = max(best_tile_score, tile_score)

        regions = extract_regions_from_map(
            anomaly_map,
            relative_threshold=relative_threshold,
            min_area=min_area,
            max_regions=max_regions_per_tile,
        )
        map_h, map_w = anomaly_map.shape[:2]
        sx = window.width / max(1.0, float(map_w))
        sy = window.height / max(1.0, float(map_h))

        mapped = []
        for candidate in regions:
            x1, y1, x2, y2 = candidate["bbox"]
            bbox = (
                int(round(window.x1 + x1 * sx)),
                int(round(window.y1 + y1 * sy)),
                int(round(window.x1 + x2 * sx)),
                int(round(window.y1 + y2 * sy)),
            )
            rank_score = tile_score * (0.50 + 0.50 * float(candidate["evidence"]))
            item = {
                "bbox": bbox,
                "rank_score": float(rank_score),
                "tile_score": tile_score,
                "evidence": float(candidate["evidence"]),
                "peak": float(candidate["peak"]),
                "area": int(candidate["area"]),
                "tile_index": int(tile_index),
                "tile_bbox": (window.x1, window.y1, window.x2, window.y2),
                "touches_tile_border": bool(candidate["touches_border"]),
                "merged_detections": 1,
            }
            mapped.append(item)
            all_regions.append(item)

        tile_results.append(
            {
                "tile_index": tile_index,
                "tile_bbox": (window.x1, window.y1, window.x2, window.y2),
                "anomaly_score": tile_score,
                "anomaly_map": anomaly_map,
                "regions": mapped,
            }
        )

    merged = _merge_overlapping_regions(all_regions, iou_threshold=merge_iou)
    if merged:
        top = float(merged[0]["rank_score"])
        merged = [r for r in merged if r["rank_score"] >= top * min_global_ratio]
        merged = merged[:max_regions]

    return {
        "image_path": str(image_path.resolve()),
        "image_size": original.size,
        "anomaly_score": float(best_tile_score),
        "regions": merged,
        "tile_results": tile_results,
        "tile_count": len(windows),
        "tile_fraction": float(tile_fraction),
        "tile_overlap": float(overlap),
    }


def save_tiled_heatmap_overlay(
    image_path: str | Path,
    tile_results: Iterable[dict],
    output_path: str | Path,
    alpha: float = 0.45,
) -> Path:
    """Compose tile anomaly maps over the complete original image."""
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    global_map = np.zeros((h, w), dtype=np.float32)

    for tile in tile_results:
        x1, y1, x2, y2 = tile["tile_bbox"]
        norm = normalize_anomaly_map(tile["anomaly_map"])
        resized = cv2.resize(norm, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
        global_map[y1:y2, x1:x2] = np.maximum(global_map[y1:y2, x1:x2], resized)

    heat = cv2.applyColorMap(np.clip(global_map * 255.0, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(base, 1.0 - alpha, heat, alpha, 0)
    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", overlay)
    if not ok:
        raise RuntimeError(f"Failed to encode heatmap overlay: {output_path}")
    encoded.tofile(str(output_path))
    return output_path


def save_regions_overlay(
    image_path: str | Path,
    regions: list[dict],
    output_path: str | Path,
    labels: list[str] | None = None,
) -> Path:
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(w, h) / 350.0)))
    font_scale = max(0.55, min(1.3, min(w, h) / 700.0))

    for i, region in enumerate(regions):
        x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        label = labels[i] if labels and i < len(labels) else f"R{i+1}"
        text = f"{label} | P={region['tile_score']:.3f}"
        ty = max(30, y1 - 8)
        cv2.putText(
            canvas,
            text,
            (max(0, x1), ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            max(1, thickness - 1),
            cv2.LINE_AA,
        )

    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode regions overlay: {output_path}")
    encoded.tofile(str(output_path))
    return output_path
