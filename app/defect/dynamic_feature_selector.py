from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class SelectorCalibration:
    cls_top1_reference: np.ndarray
    cls_margin_reference: np.ndarray
    patch_top1_reference: np.ndarray
    patch_margin_reference: np.ndarray
    cls_loo_accuracy: float
    patch_loo_accuracy: float
    support_count: int

    @staticmethod
    def _as_reference(values: Iterable[float]) -> np.ndarray:
        arr = np.asarray(list(values), dtype=np.float32).reshape(-1)
        if len(arr) == 0:
            raise ValueError("Calibration reference cannot be empty")
        if not np.isfinite(arr).all():
            raise ValueError("Calibration reference contains non-finite values")
        return np.sort(arr)

    @classmethod
    def from_loo_records(cls, records: list[dict]) -> "SelectorCalibration":
        if not records:
            raise ValueError("LOO records cannot be empty")
        return cls(
            cls_top1_reference=cls._as_reference(r["cls_top1"] for r in records),
            cls_margin_reference=cls._as_reference(r["cls_margin"] for r in records),
            patch_top1_reference=cls._as_reference(r["patch_top1"] for r in records),
            patch_margin_reference=cls._as_reference(r["patch_margin"] for r in records),
            cls_loo_accuracy=float(np.mean([r["cls_correct"] for r in records])),
            patch_loo_accuracy=float(np.mean([r["patch_correct"] for r in records])),
            support_count=len(records),
        )

    @staticmethod
    def percentile(value: float, reference: np.ndarray) -> float:
        """Empirical percentile in [0,1], calibrated only against support LOO."""
        reference = np.asarray(reference, dtype=np.float32).reshape(-1)
        if len(reference) == 0:
            raise ValueError("reference cannot be empty")
        # Mid-rank empirical CDF keeps equal values from being treated as strictly
        # more confident while avoiding arbitrary scale comparisons across methods.
        less = float(np.sum(reference < value))
        equal = float(np.sum(reference == value))
        return float((less + 0.5 * equal) / len(reference))

    def method_confidence(self, result: dict, method: str) -> dict:
        method = str(method).strip().lower()
        if method == "cls":
            top_ref = self.cls_top1_reference
            margin_ref = self.cls_margin_reference
        elif method == "patch":
            top_ref = self.patch_top1_reference
            margin_ref = self.patch_margin_reference
        else:
            raise ValueError("method must be 'cls' or 'patch'")

        top_pct = self.percentile(float(result["top1_similarity"]), top_ref)
        margin_pct = self.percentile(float(result["margin"]), margin_ref)
        confidence = 0.5 * (top_pct + margin_pct)
        return {
            "top1_percentile": top_pct,
            "margin_percentile": margin_pct,
            "confidence": confidence,
        }


class DynamicCLSPatchSelector:
    """Choose CLS or PatchMatch using support-only calibrated confidence.

    The two raw similarity spaces are not directly comparable. Each method's
    top-1 similarity and top1-top2 margin are converted to empirical percentiles
    from leave-one-out support predictions. If both methods agree, the shared
    class is accepted. If they disagree, the method with higher average
    percentile confidence wins. Exact ties are resolved by support LOO accuracy;
    if still tied, CLS is used as a deterministic fallback.
    """

    def __init__(self, calibration: SelectorCalibration):
        self.calibration = calibration

    def select(self, cls_result: dict, patch_result: dict) -> dict:
        cls_conf = self.calibration.method_confidence(cls_result, "cls")
        patch_conf = self.calibration.method_confidence(patch_result, "patch")

        cls_pred = cls_result["predicted_class"]
        patch_pred = patch_result["predicted_class"]
        agreed = cls_pred == patch_pred

        if agreed:
            selected_method = "agree"
            predicted_class = cls_pred
        else:
            delta = cls_conf["confidence"] - patch_conf["confidence"]
            if delta > 1e-12:
                selected_method = "cls"
                predicted_class = cls_pred
            elif delta < -1e-12:
                selected_method = "patch"
                predicted_class = patch_pred
            elif self.calibration.cls_loo_accuracy > self.calibration.patch_loo_accuracy:
                selected_method = "cls_tiebreak"
                predicted_class = cls_pred
            elif self.calibration.patch_loo_accuracy > self.calibration.cls_loo_accuracy:
                selected_method = "patch_tiebreak"
                predicted_class = patch_pred
            else:
                selected_method = "cls_deterministic_tiebreak"
                predicted_class = cls_pred

        return {
            "predicted_class": predicted_class,
            "selected_method": selected_method,
            "agreed": bool(agreed),
            "cls_confidence": float(cls_conf["confidence"]),
            "cls_top1_percentile": float(cls_conf["top1_percentile"]),
            "cls_margin_percentile": float(cls_conf["margin_percentile"]),
            "patch_confidence": float(patch_conf["confidence"]),
            "patch_top1_percentile": float(patch_conf["top1_percentile"]),
            "patch_margin_percentile": float(patch_conf["margin_percentile"]),
            "confidence_delta_cls_minus_patch": float(
                cls_conf["confidence"] - patch_conf["confidence"]
            ),
        }
