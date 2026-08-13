from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass
class PatchExemplar:
    label: str
    image_path: str
    tokens: np.ndarray
    weights: np.ndarray


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected [N,D] tokens, got {x.shape}")
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def _normalize_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float32).reshape(-1)
    w = np.clip(w, 0.0, None)
    total = float(w.sum())
    if total <= 1e-12:
        return np.full_like(w, 1.0 / max(1, len(w)), dtype=np.float32)
    return w / total


def select_anomaly_patch_tokens(
    patch_tokens: np.ndarray,
    anomaly_roi: np.ndarray,
    top_fraction: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Select the hottest PatchCore-guided DINOv2 tokens.

    Returns ``selected_tokens``, normalized ``selected_weights`` and selected
    flat token indices. The anomaly map is only used as a spatial selector; the
    token descriptors themselves come from frozen DINOv2.
    """
    tokens = _normalize_rows(patch_tokens)
    n_tokens = int(tokens.shape[0])
    grid = int(round(math.sqrt(n_tokens)))
    if grid * grid != n_tokens:
        raise ValueError(f"Expected a square token grid, got {n_tokens} tokens")

    fraction = float(top_fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("top_fraction must be in (0, 1]")

    anomaly = np.asarray(anomaly_roi, dtype=np.float32)
    if anomaly.ndim != 2:
        raise ValueError(f"anomaly_roi must be 2-D, got {anomaly.shape}")
    if not np.isfinite(anomaly).all():
        raise ValueError("anomaly_roi contains non-finite values")

    anomaly = np.clip(anomaly, 0.0, None)
    token_map = cv2.resize(
        anomaly,
        (grid, grid),
        interpolation=cv2.INTER_AREA if anomaly.shape[0] >= grid else cv2.INTER_LINEAR,
    ).astype(np.float32)
    flat_weights = token_map.reshape(-1)

    k = max(1, min(n_tokens, int(math.ceil(n_tokens * fraction))))
    if float(flat_weights.max()) <= 1e-12:
        # Degenerate anomaly map: fall back to the central k tokens rather than
        # inventing a semantic class-specific rule.
        yy, xx = np.mgrid[0:grid, 0:grid]
        cy = (grid - 1) / 2.0
        cx = (grid - 1) / 2.0
        distance = ((yy - cy) ** 2 + (xx - cx) ** 2).reshape(-1)
        indices = np.argsort(distance)[:k]
        chosen_weights = np.ones(k, dtype=np.float32)
    else:
        indices = np.argpartition(flat_weights, -k)[-k:]
        indices = indices[np.argsort(flat_weights[indices])[::-1]]
        chosen_weights = flat_weights[indices]

    return (
        tokens[indices].astype(np.float32),
        _normalize_weights(chosen_weights),
        indices.astype(np.int32),
    )


def bidirectional_patch_similarity(
    query_tokens: np.ndarray,
    query_weights: np.ndarray,
    support_tokens: np.ndarray,
    support_weights: np.ndarray,
) -> float:
    """Weighted symmetric Chamfer-style cosine similarity between local tokens."""
    q = _normalize_rows(query_tokens)
    s = _normalize_rows(support_tokens)
    qw = _normalize_weights(query_weights)
    sw = _normalize_weights(support_weights)

    if len(qw) != len(q) or len(sw) != len(s):
        raise ValueError("Token and weight counts must match")

    similarities = q @ s.T
    q_best = similarities.max(axis=1)
    s_best = similarities.max(axis=0)
    q_to_s = float(np.sum(qw * q_best))
    s_to_q = float(np.sum(sw * s_best))
    return 0.5 * (q_to_s + s_to_q)


class PatchExemplarMatcher:
    def __init__(self, exemplars: Iterable[PatchExemplar]):
        self.exemplars = list(exemplars)
        if not self.exemplars:
            raise ValueError("At least one patch exemplar is required")
        self.classes = sorted({e.label for e in self.exemplars})

    def class_scores(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
    ) -> tuple[dict[str, float], dict[str, str]]:
        scores: dict[str, float] = {}
        nearest: dict[str, str] = {}

        for class_name in self.classes:
            best_score = float("-inf")
            best_path = ""
            for exemplar in self.exemplars:
                if exemplar.label != class_name:
                    continue
                score = bidirectional_patch_similarity(
                    query_tokens,
                    query_weights,
                    exemplar.tokens,
                    exemplar.weights,
                )
                if score > best_score:
                    best_score = score
                    best_path = exemplar.image_path
            scores[class_name] = float(best_score)
            nearest[class_name] = best_path

        return scores, nearest

    def predict(
        self,
        query_tokens: np.ndarray,
        query_weights: np.ndarray,
    ) -> dict:
        scores, nearest = self.class_scores(query_tokens, query_weights)
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_class, top1 = ranked[0]
        if len(ranked) > 1:
            top2_class, top2 = ranked[1]
            margin = float(top1 - top2)
        else:
            top2_class, top2, margin = None, float("-inf"), float("inf")
        return {
            "predicted_class": top1_class,
            "top1_similarity": float(top1),
            "top2_class": top2_class,
            "top2_similarity": float(top2),
            "margin": margin,
            "nearest_exemplar": nearest[top1_class],
            "class_scores": scores,
        }
