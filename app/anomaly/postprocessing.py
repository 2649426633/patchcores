from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def normalize_anomaly_map(anomaly_map):
    x = np.asarray(anomaly_map, dtype=np.float32)
    min_v = float(x.min())
    max_v = float(x.max())
    if max_v - min_v < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - min_v) / (max_v - min_v)


def save_heatmap(anomaly_map, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    norm = normalize_anomaly_map(anomaly_map)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(output_path), heatmap)
    return output_path


def _prepare_regions(anomaly_map, relative_threshold=0.70):
    """Prepare a lightly denoised binary anomaly map and connected components."""
    norm = normalize_anomaly_map(anomaly_map)

    # Keep smoothing weak so small industrial defects are not erased.
    smooth = cv2.GaussianBlur(norm, (0, 0), sigmaX=0.8, sigmaY=0.8)
    binary = (smooth >= relative_threshold).astype(np.uint8)

    # Close tiny gaps inside one hot region. Do not use MORPH_OPEN here because
    # opening can remove narrow scratches or other thin defects.
    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return norm, smooth, binary, num_labels, labels, stats


def _component_bbox(stats, label):
    x = int(stats[label, cv2.CC_STAT_LEFT])
    y = int(stats[label, cv2.CC_STAT_TOP])
    w = int(stats[label, cv2.CC_STAT_WIDTH])
    h = int(stats[label, cv2.CC_STAT_HEIGHT])
    return x, y, x + w, y + h


def _distance_point_to_component(labels, label, px, py):
    ys, xs = np.where(labels == label)
    if xs.size == 0:
        return float("inf")
    dx = xs.astype(np.float32) - float(px)
    dy = ys.astype(np.float32) - float(py)
    return float(np.sqrt(np.min(dx * dx + dy * dy)))


def _candidate_regions(anomaly_map, relative_threshold=0.70, min_area=10):
    """Return connected anomaly regions ranked by anomaly evidence."""
    norm, smooth, _, num_labels, labels, stats = _prepare_regions(
        anomaly_map,
        relative_threshold=relative_threshold,
    )

    if num_labels <= 1:
        return []

    h, w = smooth.shape[:2]
    candidates = []

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = labels == label
        values = smooth[component_mask]
        if values.size == 0:
            continue

        peak = float(values.max())
        q90 = float(np.quantile(values, 0.90))
        mean = float(values.mean())
        evidence = 0.55 * peak + 0.30 * q90 + 0.15 * mean

        x1, y1, x2, y2 = _component_bbox(stats, label)
        touches_border = x1 <= 1 or y1 <= 1 or x2 >= (w - 1) or y2 >= (h - 1)
        if touches_border:
            evidence *= 0.92

        candidates.append(
            {
                "label": int(label),
                "bbox": (x1, y1, x2, y2),
                "area": area,
                "peak": peak,
                "q90": q90,
                "mean": mean,
                "touches_border": bool(touches_border),
                "evidence": float(evidence),
            }
        )

    candidates.sort(
        key=lambda item: (item["evidence"], item["peak"], item["q90"], item["area"]),
        reverse=True,
    )
    return candidates


def extract_regions_from_map(
    anomaly_map,
    relative_threshold=0.70,
    min_area=10,
    max_regions=None,
):
    """Return all meaningful PatchCore hot regions instead of forcing one bbox.

    Each returned item contains ``bbox``, ``evidence``, ``peak``, ``area`` and
    ``touches_border``.  This is intended for industrial images where one defect
    can fragment into several connected components or several anomalies can be
    present in the same image.
    """
    regions = _candidate_regions(
        anomaly_map,
        relative_threshold=relative_threshold,
        min_area=min_area,
    )
    if max_regions is not None:
        regions = regions[: max(0, int(max_regions))]
    return regions


def extract_bbox_from_map(
    anomaly_map,
    relative_threshold=0.70,
    min_area=10,
    peak_max_distance=8.0,
):
    """Extract one primary bbox using the global anomaly peak as the anchor.

    This legacy single-region API is kept for existing pipelines.  New full-image
    industrial inspection should prefer ``extract_regions_from_map`` so valid
    secondary regions are not silently discarded.
    """
    norm, _, _, num_labels, labels, stats = _prepare_regions(
        anomaly_map,
        relative_threshold=relative_threshold,
    )

    peak_index = int(np.argmax(norm))
    peak_y, peak_x = np.unravel_index(peak_index, norm.shape)

    nearest = None
    nearest_distance = float("inf")

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        if labels[peak_y, peak_x] == label:
            return _component_bbox(stats, label)

        distance = _distance_point_to_component(labels, label, peak_x, peak_y)
        if distance < nearest_distance:
            nearest_distance = distance
            nearest = label

    if nearest is not None and nearest_distance <= peak_max_distance:
        return _component_bbox(stats, nearest)

    candidates = _candidate_regions(
        anomaly_map,
        relative_threshold=relative_threshold,
        min_area=min_area,
    )
    if not candidates:
        return None
    return candidates[0]["bbox"]


def save_overlay_with_bbox(
    display_image: Image.Image,
    anomaly_map,
    output_path,
    relative_threshold=0.70,
    alpha=0.45,
):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rgb = np.asarray(display_image.convert("RGB"))
    base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    norm = normalize_anomaly_map(anomaly_map)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    if heatmap.shape[:2] != base.shape[:2]:
        heatmap = cv2.resize(
            heatmap,
            (base.shape[1], base.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    overlay = cv2.addWeighted(base, 1.0 - alpha, heatmap, alpha, 0)
    bbox = extract_bbox_from_map(anomaly_map, relative_threshold=relative_threshold)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.imwrite(str(output_path), overlay)
    return output_path, bbox
