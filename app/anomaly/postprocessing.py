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


def extract_bbox_from_map(anomaly_map, relative_threshold=0.70, min_area=10):
    norm = normalize_anomaly_map(anomaly_map)
    binary = (norm >= relative_threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None
    x, y, w, h = cv2.boundingRect(contour)
    return int(x), int(y), int(x + w), int(y + h)


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
