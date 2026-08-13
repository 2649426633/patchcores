from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from app.defect.defect_bank import DefectExemplarBank


@dataclass
class OpenSetCalibration:
    similarity_threshold: float
    margin_threshold: float
    support_quantile: float
    support_count: int
    fusion_cls_weight: float = 0.50
    fusion_center_weight: float = 0.50
    center_fraction: float = 0.50

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "similarity_threshold": float(self.similarity_threshold),
            "margin_threshold": float(self.margin_threshold),
            "support_quantile": float(self.support_quantile),
            "support_count": int(self.support_count),
            "fusion_cls_weight": float(self.fusion_cls_weight),
            "fusion_center_weight": float(self.fusion_center_weight),
            "center_fraction": float(self.center_fraction),
            "calibration_source": "support_leave_one_out_only",
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpenSetCalibration":
        return cls(
            similarity_threshold=float(data["similarity_threshold"]),
            margin_threshold=float(data["margin_threshold"]),
            support_quantile=float(data.get("support_quantile", 0.10)),
            support_count=int(data.get("support_count", 0)),
            fusion_cls_weight=float(data.get("fusion_cls_weight", 0.50)),
            fusion_center_weight=float(data.get("fusion_center_weight", 0.50)),
            center_fraction=float(data.get("center_fraction", 0.50)),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OpenSetCalibration":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def fused_class_scores(
    cls_bank: DefectExemplarBank,
    center_bank: DefectExemplarBank,
    cls_embedding: np.ndarray,
    center_embedding: np.ndarray,
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
) -> dict[str, float]:
    if cls_bank.classes != center_bank.classes:
        raise ValueError("CLS and center banks must contain identical classes")

    weight_sum = float(cls_weight + center_weight)
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    cls_weight = float(cls_weight) / weight_sum
    center_weight = float(center_weight) / weight_sum

    cls_result = cls_bank.predict_embedding(cls_embedding)
    center_result = center_bank.predict_embedding(center_embedding)

    return {
        class_name: cls_weight * cls_result["class_scores"][class_name]
        + center_weight * center_result["class_scores"][class_name]
        for class_name in cls_bank.classes
    }


def rank_scores(class_scores: dict[str, float]) -> dict:
    if not class_scores:
        raise ValueError("class_scores cannot be empty")
    ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
    top1_class, top1_score = ranked[0]
    if len(ranked) > 1:
        top2_class, top2_score = ranked[1]
        margin = float(top1_score - top2_score)
    else:
        top2_class, top2_score = None, float("-inf")
        margin = float("inf")
    return {
        "top1_class": top1_class,
        "top1_similarity": float(top1_score),
        "top2_class": top2_class,
        "top2_similarity": float(top2_score),
        "margin": margin,
        "class_scores": class_scores,
    }


def calibrate_from_support(
    cls_embeddings: np.ndarray,
    center_embeddings: np.ndarray,
    labels: Iterable[str],
    support_quantile: float = 0.10,
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
    center_fraction: float = 0.50,
) -> tuple[OpenSetCalibration, list[dict]]:
    """Calibrate Unknown thresholds using support exemplars only.

    Every support exemplar is treated as a pseudo-query once. Its own exemplar is
    removed, class scores are computed from the remaining support set, and the
    true-class similarity plus true-vs-best-other margin are recorded. No query
    or test labels are used to derive thresholds.
    """

    labels = list(labels)
    cls_embeddings = _normalize_rows(cls_embeddings)
    center_embeddings = _normalize_rows(center_embeddings)

    if cls_embeddings.shape != center_embeddings.shape:
        raise ValueError("CLS and center support embeddings must have identical shape")
    if len(labels) != len(cls_embeddings):
        raise ValueError("labels length must match support embeddings")
    if not (0.0 <= support_quantile < 0.5):
        raise ValueError("support_quantile must be in [0, 0.5)")

    unique_classes = sorted(set(labels))
    for class_name in unique_classes:
        if labels.count(class_name) < 2:
            raise ValueError(
                f"Class {class_name!r} needs at least 2 support exemplars for leave-one-out calibration"
            )

    weight_sum = float(cls_weight + center_weight)
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    cls_weight = float(cls_weight) / weight_sum
    center_weight = float(center_weight) / weight_sum

    diagnostics = []
    true_scores = []
    true_margins = []

    for query_index, true_class in enumerate(labels):
        cls_query = cls_embeddings[query_index]
        center_query = center_embeddings[query_index]

        class_scores = {}
        for class_name in unique_classes:
            indices = [
                i
                for i, label in enumerate(labels)
                if label == class_name and i != query_index
            ]
            if not indices:
                continue
            cls_score = float(np.max(cls_embeddings[indices] @ cls_query))
            center_score = float(np.max(center_embeddings[indices] @ center_query))
            class_scores[class_name] = cls_weight * cls_score + center_weight * center_score

        if true_class not in class_scores:
            raise RuntimeError(f"No remaining same-class support for {true_class!r}")

        true_score = float(class_scores[true_class])
        other_scores = [score for name, score in class_scores.items() if name != true_class]
        best_other = max(other_scores) if other_scores else float("-inf")
        true_margin = float(true_score - best_other) if np.isfinite(best_other) else float("inf")
        ranked = rank_scores(class_scores)

        true_scores.append(true_score)
        if np.isfinite(true_margin):
            true_margins.append(true_margin)

        diagnostics.append(
            {
                "support_index": query_index,
                "true_class": true_class,
                "loo_predicted_class": ranked["top1_class"],
                "loo_correct": int(ranked["top1_class"] == true_class),
                "true_class_similarity": true_score,
                "best_other_similarity": best_other,
                "true_class_margin": true_margin,
                "top1_similarity": ranked["top1_similarity"],
                "top1_margin": ranked["margin"],
            }
        )

    similarity_threshold = float(np.quantile(true_scores, support_quantile))
    if true_margins:
        margin_threshold = max(0.0, float(np.quantile(true_margins, support_quantile)))
    else:
        margin_threshold = 0.0

    calibration = OpenSetCalibration(
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        support_quantile=float(support_quantile),
        support_count=len(labels),
        fusion_cls_weight=cls_weight,
        fusion_center_weight=center_weight,
        center_fraction=float(center_fraction),
    )
    return calibration, diagnostics


class FusedOpenSetRecognizer:
    def __init__(
        self,
        cls_bank: DefectExemplarBank,
        center_bank: DefectExemplarBank,
        calibration: OpenSetCalibration,
    ):
        if cls_bank.classes != center_bank.classes:
            raise ValueError("CLS and center banks must contain identical classes")
        self.cls_bank = cls_bank
        self.center_bank = center_bank
        self.calibration = calibration

    def predict_embeddings(
        self,
        cls_embedding: np.ndarray,
        center_embedding: np.ndarray,
    ) -> dict:
        scores = fused_class_scores(
            self.cls_bank,
            self.center_bank,
            cls_embedding,
            center_embedding,
            cls_weight=self.calibration.fusion_cls_weight,
            center_weight=self.calibration.fusion_center_weight,
        )
        ranked = rank_scores(scores)

        similarity_ok = ranked["top1_similarity"] >= self.calibration.similarity_threshold
        margin_ok = ranked["margin"] >= self.calibration.margin_threshold
        accepted = bool(similarity_ok and margin_ok)

        return {
            "predicted_class": ranked["top1_class"] if accepted else "Unknown",
            "nearest_known_class": ranked["top1_class"],
            "accepted_as_known": accepted,
            "top1_similarity": ranked["top1_similarity"],
            "top2_class": ranked["top2_class"],
            "top2_similarity": ranked["top2_similarity"],
            "margin": ranked["margin"],
            "similarity_threshold": self.calibration.similarity_threshold,
            "margin_threshold": self.calibration.margin_threshold,
            "similarity_ok": similarity_ok,
            "margin_ok": margin_ok,
            "class_scores": ranked["class_scores"],
        }
