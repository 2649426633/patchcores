from pathlib import Path
from typing import Optional

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


def _candidate_regions(anomaly_map, relative_threshold=0.70, min_area=10):
    """Return connected anomaly regions ranked by anomaly evidence.

    The previous implementation selected the largest contour. That can prefer a
    broad background/border response over a smaller but much hotter real defect.
    Here we lightly denoise the thresholded map and rank every valid component by
    peak/high-percentile/mean anomaly strength. Border-touching components receive
    only a small penalty instead of being discarded, so real edge defects remain
    detectable.
    """
    norm = normalize_anomaly_map(anomaly_map)

    # PatchCore already smooths the anomaly map; this very small extra blur only
    # suppresses isolated pixel-scale spikes before connected-component analysis.
    smooth = cv2.GaussianBlur(norm, (0, 0), sigmaX=1.0, sigmaY=1.0)
    binary = (smooth >= relative_threshold).astype(np.uint8) * 255

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    h, w = smooth.shape[:2]
    candidates = []

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(contour)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [contour], contourIdx=-1, color=1, thickness=-1)
        values = smooth[mask.astype(bool)]
        if values.size == 0:
            continue

        peak = float(values.max())
        q90 = float(np.quantile(values, 0.90))
        mean = float(values.mean())

        # Prefer genuinely hot regions over merely large ones.
        evidence = 0.55 * peak + 0.30 * q90 + 0.15 * mean

        touches_border = x <= 1 or y <= 1 or (x + bw) >= (w - 1) or (y + bh) >= (h - 1)
        if touches_border:
            evidence *= 0.92

        candidates.append(
            {
                "bbox": (int(x), int(y), int(x + bw), int(y + bh)),
                "area": area,
                "peak": peak,
                "q90": q90,
                "mean": mean,
                "touches_border": touches_border,
                "evidence": evidence,
            }
        )

    candidates.sort(
        key=lambda item: (item["evidence"], item["peak"], item["q90"], item["area"]),
        reverse=True,
    )
    return candidates


def extract_bbox_from_map(anomaly_map, relative_threshold=0.70, min_area=10):
    """Extract the strongest candidate defect box from an anomaly map.

    This threshold is for localization visualization only. It is not the final
    PASS/NG decision threshold.
    """
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
