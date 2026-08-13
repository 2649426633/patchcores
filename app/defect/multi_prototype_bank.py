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


def _spherical_kmeans(x: np.ndarray, k: int, max_iter: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic spherical k-means for small support sets.

    Initialization is deterministic farthest-point seeding in cosine space:
    the first center is the most central sample, and each next center is the
    sample least similar to its nearest existing center. Cluster centers are
    normalized means, matching cosine-similarity inference.
    """
    x = _normalize_rows(x)
    n = len(x)
    if n == 0:
        raise ValueError("Cannot cluster an empty class")
    k = max(1, min(int(k), n))

    if k == 1:
        center = _normalize_vector(x.mean(axis=0))[None, :]
        return center.astype(np.float32), np.zeros(n, dtype=np.int32)

    pairwise = x @ x.T
    first = int(np.argmax(pairwise.mean(axis=1)))
    seed_indices = [first]
    while len(seed_indices) < k:
        nearest = np.max(pairwise[:, seed_indices], axis=1)
        nearest[seed_indices] = np.inf
        seed_indices.append(int(np.argmin(nearest)))

    centers = x[seed_indices].copy()
    labels = np.full(n, -1, dtype=np.int32)

    for _ in range(max_iter):
        similarities = x @ centers.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        new_centers = []
        for cluster_index in range(k):
            members = x[labels == cluster_index]
            if len(members) == 0:
                # Deterministic recovery: choose sample least represented by any
                # currently non-empty center.
                nearest = np.max(similarities, axis=1)
                replacement = x[int(np.argmin(nearest))]
                new_centers.append(replacement)
            else:
                new_centers.append(_normalize_vector(members.mean(axis=0)))
        centers = np.stack(new_centers, axis=0).astype(np.float32)

    return _normalize_rows(centers), labels


@dataclass
class ClassPrototypeInfo:
    class_name: str
    cluster_sizes: list[int]


class MultiPrototypeDefectBank:
    """Represent each class with multiple cosine-space prototypes."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: Iterable[str],
        prototypes_per_class: int = 2,
    ):
        self.embeddings = _normalize_rows(embeddings)
        self.labels = list(labels)
        if len(self.labels) != len(self.embeddings):
            raise ValueError("labels length must match embeddings")
        if not self.labels:
            raise ValueError("bank cannot be empty")
        if prototypes_per_class <= 0:
            raise ValueError("prototypes_per_class must be positive")

        self.prototypes_per_class = int(prototypes_per_class)
        self.classes = sorted(set(self.labels))
        self.prototypes: dict[str, np.ndarray] = {}
        self.prototype_info: dict[str, ClassPrototypeInfo] = {}

        for class_name in self.classes:
            indices = [i for i, label in enumerate(self.labels) if label == class_name]
            class_embeddings = self.embeddings[indices]
            centers, assignments = _spherical_kmeans(
                class_embeddings,
                k=min(self.prototypes_per_class, len(class_embeddings)),
            )
            sizes = [int(np.sum(assignments == i)) for i in range(len(centers))]
            self.prototypes[class_name] = centers.astype(np.float32)
            self.prototype_info[class_name] = ClassPrototypeInfo(
                class_name=class_name,
                cluster_sizes=sizes,
            )

    def predict_embedding(self, embedding: np.ndarray) -> dict:
        query = _normalize_vector(embedding)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: {query.shape[0]} vs {self.embeddings.shape[1]}"
            )

        class_scores: dict[str, float] = {}
        best_prototype_index: dict[str, int] = {}
        for class_name in self.classes:
            sims = self.prototypes[class_name] @ query
            best = int(np.argmax(sims))
            class_scores[class_name] = float(sims[best])
            best_prototype_index[class_name] = best

        ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
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
            "best_prototype_index": best_prototype_index[top1_class],
            "class_scores": class_scores,
        }
