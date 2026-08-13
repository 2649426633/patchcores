from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.defect.open_set_fusion import OpenSetCalibration, rank_scores


@dataclass
class PseudoUnknownCalibrationSummary:
    known_accept_rate: float
    pseudo_unknown_reject_rate: float
    balanced_accuracy: float
    min_side_rate: float
    similarity_threshold: float
    margin_threshold: float
    support_count: int

    def to_dict(self) -> dict:
        return {
            "calibration_source": "support_known_vs_class_held_out_pseudo_unknown_only",
            "known_accept_rate": float(self.known_accept_rate),
            "pseudo_unknown_reject_rate": float(self.pseudo_unknown_reject_rate),
            "balanced_accuracy": float(self.balanced_accuracy),
            "min_side_rate": float(self.min_side_rate),
            "similarity_threshold": float(self.similarity_threshold),
            "margin_threshold": float(self.margin_threshold),
            "support_count": int(self.support_count),
        }


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D embeddings, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _fused_scores_from_indices(
    cls_embeddings: np.ndarray,
    center_embeddings: np.ndarray,
    labels: list[str],
    query_index: int,
    allowed_indices: list[int],
    cls_weight: float,
    center_weight: float,
) -> dict[str, float]:
    cls_query = cls_embeddings[query_index]
    center_query = center_embeddings[query_index]
    classes = sorted({labels[i] for i in allowed_indices})
    scores = {}
    for class_name in classes:
        indices = [i for i in allowed_indices if labels[i] == class_name]
        if not indices:
            continue
        cls_score = float(np.max(cls_embeddings[indices] @ cls_query))
        center_score = float(np.max(center_embeddings[indices] @ center_query))
        scores[class_name] = cls_weight * cls_score + center_weight * center_score
    return scores


def build_support_calibration_records(
    cls_embeddings: np.ndarray,
    center_embeddings: np.ndarray,
    labels: Iterable[str],
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
) -> list[dict]:
    """Create known and pseudo-unknown calibration records from support only.

    For every support exemplar two pseudo-query situations are produced:
    1. known: remove only the query exemplar; its class remains in the bank.
    2. pseudo_unknown: remove the query exemplar's entire class from the bank.

    This mirrors leave-one-defect-class-out deployment without using query/test
    samples to select thresholds.
    """

    labels = list(labels)
    cls_embeddings = _normalize_rows(cls_embeddings)
    center_embeddings = _normalize_rows(center_embeddings)
    if cls_embeddings.shape != center_embeddings.shape:
        raise ValueError("CLS and center embeddings must have identical shape")
    if len(labels) != len(cls_embeddings):
        raise ValueError("labels length must match embeddings")

    unique_classes = sorted(set(labels))
    if len(unique_classes) < 2:
        raise ValueError("At least two defect classes are required")
    for class_name in unique_classes:
        if labels.count(class_name) < 2:
            raise ValueError(
                f"Class {class_name!r} needs at least 2 support exemplars"
            )

    weight_sum = float(cls_weight + center_weight)
    if weight_sum <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    cls_weight = float(cls_weight) / weight_sum
    center_weight = float(center_weight) / weight_sum

    records: list[dict] = []
    all_indices = list(range(len(labels)))

    for query_index, true_class in enumerate(labels):
        known_indices = [i for i in all_indices if i != query_index]
        known_scores = _fused_scores_from_indices(
            cls_embeddings,
            center_embeddings,
            labels,
            query_index,
            known_indices,
            cls_weight,
            center_weight,
        )
        known_ranked = rank_scores(known_scores)
        records.append(
            {
                "support_index": query_index,
                "true_class": true_class,
                "scenario": "known",
                "desired_known": 1,
                "predicted_class": known_ranked["top1_class"],
                "classification_correct": int(known_ranked["top1_class"] == true_class),
                "top1_similarity": float(known_ranked["top1_similarity"]),
                "margin": float(known_ranked["margin"]),
            }
        )

        unknown_indices = [i for i in all_indices if labels[i] != true_class]
        unknown_scores = _fused_scores_from_indices(
            cls_embeddings,
            center_embeddings,
            labels,
            query_index,
            unknown_indices,
            cls_weight,
            center_weight,
        )
        unknown_ranked = rank_scores(unknown_scores)
        records.append(
            {
                "support_index": query_index,
                "true_class": true_class,
                "scenario": "pseudo_unknown",
                "desired_known": 0,
                "predicted_class": unknown_ranked["top1_class"],
                "classification_correct": 0,
                "top1_similarity": float(unknown_ranked["top1_similarity"]),
                "margin": float(unknown_ranked["margin"]),
            }
        )

    return records


def select_pseudo_unknown_thresholds(
    records: list[dict],
) -> PseudoUnknownCalibrationSummary:
    if not records:
        raise ValueError("records cannot be empty")

    known = [r for r in records if r["scenario"] == "known"]
    unknown = [r for r in records if r["scenario"] == "pseudo_unknown"]
    if not known or not unknown:
        raise ValueError("Both known and pseudo_unknown records are required")

    similarities = [float(r["top1_similarity"]) for r in records]
    margins = [max(0.0, float(r["margin"])) for r in records]

    sim_min = min(similarities)
    similarity_candidates = sorted(
        set([sim_min - 1e-6, *similarities])
    )
    margin_candidates = sorted(set([0.0, *margins]))

    best = None
    for sim_threshold in similarity_candidates:
        for margin_threshold in margin_candidates:
            known_accept = np.mean(
                [
                    float(r["top1_similarity"]) >= sim_threshold
                    and float(r["margin"]) >= margin_threshold
                    for r in known
                ]
            )
            unknown_reject = np.mean(
                [
                    not (
                        float(r["top1_similarity"]) >= sim_threshold
                        and float(r["margin"]) >= margin_threshold
                    )
                    for r in unknown
                ]
            )
            balanced = 0.5 * (known_accept + unknown_reject)
            min_side = min(known_accept, unknown_reject)

            # Primary goal: balanced known/unknown behavior. Tie-break toward a
            # stronger worst side and then toward preserving known acceptance.
            key = (balanced, min_side, known_accept, unknown_reject)
            if best is None or key > best[0]:
                best = (
                    key,
                    float(sim_threshold),
                    float(margin_threshold),
                    float(known_accept),
                    float(unknown_reject),
                )

    assert best is not None
    _, sim_threshold, margin_threshold, known_accept, unknown_reject = best
    return PseudoUnknownCalibrationSummary(
        known_accept_rate=known_accept,
        pseudo_unknown_reject_rate=unknown_reject,
        balanced_accuracy=0.5 * (known_accept + unknown_reject),
        min_side_rate=min(known_accept, unknown_reject),
        similarity_threshold=sim_threshold,
        margin_threshold=margin_threshold,
        support_count=len(known),
    )


def calibrate_with_pseudo_unknown(
    cls_embeddings: np.ndarray,
    center_embeddings: np.ndarray,
    labels: Iterable[str],
    cls_weight: float = 0.50,
    center_weight: float = 0.50,
    center_fraction: float = 0.50,
) -> tuple[OpenSetCalibration, list[dict], PseudoUnknownCalibrationSummary]:
    records = build_support_calibration_records(
        cls_embeddings=cls_embeddings,
        center_embeddings=center_embeddings,
        labels=labels,
        cls_weight=cls_weight,
        center_weight=center_weight,
    )
    summary = select_pseudo_unknown_thresholds(records)

    weight_sum = float(cls_weight + center_weight)
    cls_weight = float(cls_weight) / weight_sum
    center_weight = float(center_weight) / weight_sum

    calibration = OpenSetCalibration(
        similarity_threshold=summary.similarity_threshold,
        margin_threshold=summary.margin_threshold,
        support_quantile=-1.0,
        support_count=summary.support_count,
        fusion_cls_weight=cls_weight,
        fusion_center_weight=center_weight,
        center_fraction=float(center_fraction),
    )
    return calibration, records, summary
