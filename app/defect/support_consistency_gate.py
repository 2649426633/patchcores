from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _normalize_vector(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def fused_exemplar_similarities(
    cls_query: np.ndarray,
    center_query: np.ndarray,
    cls_support: np.ndarray,
    center_support: np.ndarray,
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
) -> np.ndarray:
    """Return one fused cosine similarity per support exemplar."""
    cls_support = _normalize_rows(cls_support)
    center_support = _normalize_rows(center_support)
    if cls_support.shape != center_support.shape:
        raise ValueError("CLS and center support embeddings must have identical shape")

    cls_query = _normalize_vector(cls_query)
    center_query = _normalize_vector(center_query)
    if cls_query.shape[0] != cls_support.shape[1]:
        raise ValueError("CLS embedding dimension mismatch")
    if center_query.shape[0] != center_support.shape[1]:
        raise ValueError("Center embedding dimension mismatch")

    weight_sum = float(cls_weight + center_weight)
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    cls_weight = float(cls_weight) / weight_sum
    center_weight = float(center_weight) / weight_sum

    return (
        cls_weight * (cls_support @ cls_query)
        + center_weight * (center_support @ center_query)
    ).astype(np.float32)


def class_max_scores(similarities: np.ndarray, labels: Iterable[str]) -> dict[str, float]:
    labels = list(labels)
    similarities = np.asarray(similarities, dtype=np.float32).reshape(-1)
    if len(labels) != len(similarities):
        raise ValueError("labels length must match similarities")

    scores: dict[str, float] = {}
    for class_name in sorted(set(labels)):
        indices = [i for i, label in enumerate(labels) if label == class_name]
        scores[class_name] = float(np.max(similarities[indices]))
    return scores


def topk_class_consistency(
    similarities: np.ndarray,
    labels: Iterable[str],
    class_name: str,
    top_k: int = 3,
) -> float:
    """Mean of the strongest K exemplar similarities inside one class."""
    labels = list(labels)
    similarities = np.asarray(similarities, dtype=np.float32).reshape(-1)
    if len(labels) != len(similarities):
        raise ValueError("labels length must match similarities")
    if top_k <= 0:
        raise ValueError("top_k must be positive")

    values = np.asarray(
        [similarities[i] for i, label in enumerate(labels) if label == class_name],
        dtype=np.float32,
    )
    if len(values) == 0:
        raise ValueError(f"No support exemplars for class {class_name!r}")
    k = min(int(top_k), len(values))
    strongest = np.partition(values, len(values) - k)[-k:]
    return float(np.mean(strongest))


@dataclass
class SupportConsistencyCalibration:
    class_thresholds: dict[str, float]
    top_k: int
    support_quantile: float
    support_count: int
    cls_weight: float = 0.50
    center_weight: float = 0.50


def calibrate_support_consistency(
    cls_embeddings: np.ndarray,
    center_embeddings: np.ndarray,
    labels: Iterable[str],
    top_k: int = 3,
    support_quantile: float = 0.10,
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
) -> tuple[SupportConsistencyCalibration, list[dict]]:
    """Calibrate class-conditional density thresholds from support LOO only.

    Each support exemplar is removed once. Its fused similarities to the remaining
    support set are computed, and the mean of the top-K similarities within its
    *true class* is recorded. Each class gets its own lower-tail quantile threshold.
    No evaluation/query labels are used to derive these thresholds.
    """
    labels = list(labels)
    cls_embeddings = _normalize_rows(cls_embeddings)
    center_embeddings = _normalize_rows(center_embeddings)
    if cls_embeddings.shape != center_embeddings.shape:
        raise ValueError("CLS and center support embeddings must have identical shape")
    if len(labels) != len(cls_embeddings):
        raise ValueError("labels length must match support embeddings")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not (0.0 <= support_quantile < 0.5):
        raise ValueError("support_quantile must be in [0,0.5)")

    classes = sorted(set(labels))
    for class_name in classes:
        if labels.count(class_name) < 2:
            raise ValueError(
                f"Class {class_name!r} needs at least two support exemplars for LOO calibration"
            )

    diagnostics: list[dict] = []
    per_class_values: dict[str, list[float]] = {c: [] for c in classes}

    for query_index, true_class in enumerate(labels):
        keep = [i for i in range(len(labels)) if i != query_index]
        keep_labels = [labels[i] for i in keep]
        similarities = fused_exemplar_similarities(
            cls_embeddings[query_index],
            center_embeddings[query_index],
            cls_embeddings[keep],
            center_embeddings[keep],
            cls_weight=cls_weight,
            center_weight=center_weight,
        )
        scores = class_max_scores(similarities, keep_labels)
        predicted_class = max(scores.items(), key=lambda item: item[1])[0]
        consistency = topk_class_consistency(
            similarities,
            keep_labels,
            true_class,
            top_k=top_k,
        )
        per_class_values[true_class].append(consistency)
        diagnostics.append(
            {
                "support_index": query_index,
                "true_class": true_class,
                "loo_predicted_class": predicted_class,
                "loo_correct": int(predicted_class == true_class),
                "true_class_topk_consistency": consistency,
            }
        )

    class_thresholds = {
        class_name: float(np.quantile(values, support_quantile))
        for class_name, values in per_class_values.items()
    }
    calibration = SupportConsistencyCalibration(
        class_thresholds=class_thresholds,
        top_k=int(top_k),
        support_quantile=float(support_quantile),
        support_count=len(labels),
        cls_weight=float(cls_weight),
        center_weight=float(center_weight),
    )
    return calibration, diagnostics


class SupportConsistencyGate:
    """Additional Unknown gate based on same-class multi-exemplar support density."""

    def __init__(
        self,
        cls_embeddings: np.ndarray,
        center_embeddings: np.ndarray,
        labels: Iterable[str],
        calibration: SupportConsistencyCalibration,
    ):
        self.cls_embeddings = _normalize_rows(cls_embeddings)
        self.center_embeddings = _normalize_rows(center_embeddings)
        self.labels = list(labels)
        if self.cls_embeddings.shape != self.center_embeddings.shape:
            raise ValueError("CLS and center support embeddings must have identical shape")
        if len(self.labels) != len(self.cls_embeddings):
            raise ValueError("labels length must match support embeddings")
        self.calibration = calibration

    def evaluate(
        self,
        cls_embedding: np.ndarray,
        center_embedding: np.ndarray,
        predicted_class: str,
    ) -> dict:
        similarities = fused_exemplar_similarities(
            cls_embedding,
            center_embedding,
            self.cls_embeddings,
            self.center_embeddings,
            cls_weight=self.calibration.cls_weight,
            center_weight=self.calibration.center_weight,
        )
        consistency = topk_class_consistency(
            similarities,
            self.labels,
            predicted_class,
            top_k=self.calibration.top_k,
        )
        threshold = float(self.calibration.class_thresholds[predicted_class])
        return {
            "predicted_class": predicted_class,
            "consistency_score": float(consistency),
            "consistency_threshold": threshold,
            "consistency_ok": bool(consistency >= threshold),
        }
