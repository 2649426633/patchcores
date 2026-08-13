from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from app.anomaly.patchcore_adapter import PatchCoreAdapter
from app.anomaly.postprocessing import extract_bbox_from_map, normalize_anomaly_map
from app.anomaly.preprocessing import load_display_image, map_display_bbox_to_original
from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import DINOv2Adapter


class PatchCoreDINOv2Pipeline:
    """End-to-end industrial defect pipeline.

    PatchCore answers where the anomaly is. The predicted anomaly ROI is then
    cropped from the same preprocessed image coordinate system and passed to a
    frozen DINOv2 backbone.
    """

    def __init__(
        self,
        patchcore_model_dir: str | Path,
        bank_dir: Optional[str | Path] = None,
        device: Optional[str] = None,
        bbox_relative_threshold: float = 0.80,
        roi_margin: float = 0.50,
        fallback_side_ratio: float = 0.18,
        dinov2_feature_mode: str = "cls",
        center_fraction: float = 0.50,
    ):
        self.patchcore_model_dir = Path(patchcore_model_dir)
        self.bank_dir = Path(bank_dir) if bank_dir is not None else None
        self.device = device
        self.bbox_relative_threshold = float(bbox_relative_threshold)
        self.roi_margin = float(roi_margin)
        self.fallback_side_ratio = float(fallback_side_ratio)
        self.dinov2_feature_mode = str(dinov2_feature_mode).strip().lower()
        self.center_fraction = float(center_fraction)

        self.patchcore: Optional[PatchCoreAdapter] = None
        self.dinov2: Optional[DINOv2Adapter] = None
        self.bank: Optional[DefectExemplarBank] = None

    def load(self) -> None:
        self.patchcore = PatchCoreAdapter(device=self.device)
        self.patchcore.load(self.patchcore_model_dir)

        self.dinov2 = DINOv2Adapter(device=self.device)
        self.dinov2.load()

        if self.bank_dir is not None:
            self.bank = DefectExemplarBank.load(self.bank_dir)

    @staticmethod
    def _border_fill(image: Image.Image) -> tuple[int, int, int]:
        arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
        border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]], axis=0)
        median = np.median(border.astype(np.float32), axis=0)
        return tuple(int(v) for v in median)

    @staticmethod
    def _map_bbox_to_image(
        bbox: tuple[int, int, int, int],
        map_shape: tuple[int, int],
        image_size: tuple[int, int],
    ) -> tuple[int, int, int, int]:
        map_h, map_w = map_shape
        image_w, image_h = image_size
        x1, y1, x2, y2 = bbox
        sx = image_w / max(1.0, float(map_w))
        sy = image_h / max(1.0, float(map_h))
        return (
            int(round(x1 * sx)),
            int(round(y1 * sy)),
            int(round(x2 * sx)),
            int(round(y2 * sy)),
        )

    @staticmethod
    def _peak_fallback_bbox(
        anomaly_map: np.ndarray,
        image_size: tuple[int, int],
        side_ratio: float,
    ) -> tuple[int, int, int, int]:
        norm = normalize_anomaly_map(anomaly_map)
        peak_index = int(np.argmax(norm))
        peak_y, peak_x = np.unravel_index(peak_index, norm.shape)

        image_w, image_h = image_size
        px = (float(peak_x) + 0.5) * image_w / max(1, norm.shape[1])
        py = (float(peak_y) + 0.5) * image_h / max(1, norm.shape[0])
        side = max(8, int(round(min(image_w, image_h) * max(0.02, side_ratio))))
        half = side / 2.0
        return (
            int(round(px - half)),
            int(round(py - half)),
            int(round(px + half)),
            int(round(py + half)),
        )

    def _crop_square_with_margin(
        self,
        image: Image.Image,
        bbox: tuple[int, int, int, int],
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        image = image.convert("RGB")
        image_w, image_h = image.size
        x1, y1, x2, y2 = bbox

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        side = int(
            np.ceil(
                max(
                    bw * (1.0 + 2.0 * self.roi_margin),
                    bh * (1.0 + 2.0 * self.roi_margin),
                )
            )
        )
        side = max(8, side)

        left = int(np.floor(cx - side / 2.0))
        top = int(np.floor(cy - side / 2.0))
        right = left + side
        bottom = top + side

        fill = self._border_fill(image)
        roi = Image.new("RGB", (side, side), fill)

        src_left = max(0, left)
        src_top = max(0, top)
        src_right = min(image_w, right)
        src_bottom = min(image_h, bottom)

        if src_right > src_left and src_bottom > src_top:
            roi.paste(
                image.crop((src_left, src_top, src_right, src_bottom)),
                (src_left - left, src_top - top),
            )

        return roi, (left, top, right, bottom)

    @staticmethod
    def _crop_map_with_bbox(
        map_image: np.ndarray,
        crop_bbox: tuple[int, int, int, int],
    ) -> np.ndarray:
        x1, y1, x2, y2 = crop_bbox
        side_w = max(1, x2 - x1)
        side_h = max(1, y2 - y1)
        out = np.zeros((side_h, side_w), dtype=np.float32)

        h, w = map_image.shape[:2]
        src_left = max(0, x1)
        src_top = max(0, y1)
        src_right = min(w, x2)
        src_bottom = min(h, y2)

        if src_right > src_left and src_bottom > src_top:
            dst_left = src_left - x1
            dst_top = src_top - y1
            dst_right = dst_left + (src_right - src_left)
            dst_bottom = dst_top + (src_bottom - src_top)
            out[dst_top:dst_bottom, dst_left:dst_right] = map_image[
                src_top:src_bottom, src_left:src_right
            ]
        return out

    def extract_roi(self, image_path: str | Path) -> dict:
        if self.patchcore is None:
            raise RuntimeError("Pipeline is not loaded. Call load() first.")

        image_path = Path(image_path)
        patch_result = self.patchcore.predict(image_path)
        anomaly_map = np.asarray(patch_result["anomaly_map"], dtype=np.float32)

        display_image = load_display_image(
            image_path,
            resize=self.patchcore.config.resize,
            imagesize=self.patchcore.config.imagesize,
        ).convert("RGB")

        raw_bbox = extract_bbox_from_map(
            anomaly_map,
            relative_threshold=self.bbox_relative_threshold,
        )

        if raw_bbox is None:
            image_bbox = self._peak_fallback_bbox(
                anomaly_map,
                image_size=display_image.size,
                side_ratio=self.fallback_side_ratio,
            )
            bbox_source = "peak_fallback"
        else:
            image_bbox = self._map_bbox_to_image(
                raw_bbox,
                map_shape=anomaly_map.shape[:2],
                image_size=display_image.size,
            )
            bbox_source = "patchcore_component"

        original_bbox = map_display_bbox_to_original(
            image_path,
            image_bbox,
            resize=self.patchcore.config.resize,
            imagesize=self.patchcore.config.imagesize,
        )

        roi, expanded_bbox = self._crop_square_with_margin(display_image, image_bbox)

        display_norm = normalize_anomaly_map(anomaly_map)
        if display_norm.shape != (display_image.height, display_image.width):
            display_norm = cv2.resize(
                display_norm,
                display_image.size,
                interpolation=cv2.INTER_LINEAR,
            )
        anomaly_roi = self._crop_map_with_bbox(display_norm, expanded_bbox)
        anomaly_roi = normalize_anomaly_map(anomaly_roi)

        return {
            "image_path": str(image_path),
            "anomaly_score": float(patch_result["anomaly_score"]),
            "anomaly_map": anomaly_map,
            "anomaly_roi": anomaly_roi,
            "bbox": tuple(int(v) for v in image_bbox),
            "original_bbox": tuple(int(v) for v in original_bbox),
            "expanded_bbox": tuple(int(v) for v in expanded_bbox),
            "bbox_source": bbox_source,
            "display_image": display_image,
            "roi": roi,
        }

    def embed_roi(
        self,
        roi: Image.Image,
        feature_mode: Optional[str] = None,
        center_fraction: Optional[float] = None,
        spatial_weights: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if self.dinov2 is None:
            raise RuntimeError("Pipeline is not loaded. Call load() first.")
        return self.dinov2.embed(
            roi,
            feature_mode=(self.dinov2_feature_mode if feature_mode is None else feature_mode),
            center_fraction=(self.center_fraction if center_fraction is None else center_fraction),
            spatial_weights=spatial_weights,
        )

    def classify(self, image_path: str | Path) -> dict:
        if self.bank is None:
            raise RuntimeError("No defect bank loaded. Pass bank_dir when creating the pipeline.")

        roi_result = self.extract_roi(image_path)
        mode = self.dinov2_feature_mode
        embedding = self.embed_roi(
            roi_result["roi"],
            feature_mode=mode,
            spatial_weights=(roi_result["anomaly_roi"] if mode == "patch_weighted" else None),
        )
        classification = self.bank.predict_embedding(embedding)
        return {**roi_result, **classification, "embedding": embedding}

    @staticmethod
    def save_bbox_overlay(
        display_image: Image.Image,
        bbox: tuple[int, int, int, int],
        output_path: str | Path,
    ) -> Path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rgb = np.asarray(display_image.convert("RGB"), dtype=np.uint8)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        x1, y1, x2, y2 = bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.imwrite(str(output_path), canvas)
        return output_path

    @staticmethod
    def save_full_image_overlay(
        image_path: str | Path,
        bbox: tuple[int, int, int, int],
        output_path: str | Path,
        label: Optional[str] = None,
        anomaly_score: Optional[float] = None,
        similarity: Optional[float] = None,
    ) -> Path:
        """Save the ORIGINAL full-resolution image with anomaly bbox and result text."""
        image_path = Path(image_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = canvas.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in bbox]
        x1 = max(0, min(w - 1, x1))
        y1 = max(0, min(h - 1, y1))
        x2 = max(x1 + 1, min(w - 1, x2))
        y2 = max(y1 + 1, min(h - 1, y2))

        thickness = max(2, int(round(min(w, h) / 350.0)))
        font_scale = max(0.5, min(1.2, min(w, h) / 700.0))
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness)

        parts = []
        if label:
            parts.append(str(label))
        if anomaly_score is not None:
            parts.append(f"PatchCore={float(anomaly_score):.3f}")
        if similarity is not None:
            parts.append(f"sim={float(similarity):.3f}")
        text = " | ".join(parts)

        if text:
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_thickness = max(1, thickness - 1)
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
            tx = max(0, min(w - tw - 2, x1))
            ty = y1 - 8
            if ty - th - baseline < 0:
                ty = min(h - baseline - 2, y2 + th + baseline + 8)
            bg_y1 = max(0, ty - th - baseline - 4)
            bg_y2 = min(h - 1, ty + baseline + 3)
            cv2.rectangle(canvas, (tx, bg_y1), (min(w - 1, tx + tw + 6), bg_y2), (0, 0, 0), -1)
            cv2.putText(
                canvas,
                text,
                (tx + 3, ty),
                font,
                font_scale,
                (255, 255, 255),
                text_thickness,
                cv2.LINE_AA,
            )

        ext = output_path.suffix.lower() or ".jpg"
        ok, encoded = cv2.imencode(ext, canvas)
        if not ok:
            raise RuntimeError(f"Failed to encode marked image: {output_path}")
        encoded.tofile(str(output_path))
        return output_path
