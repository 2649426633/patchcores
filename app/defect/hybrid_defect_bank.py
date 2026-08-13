from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class HybridDefectBank:
    """Few-shot defect bank combining nearest exemplar and class prototype.

    All embeddings are L2-normalized. For each class we compute:
      exemplar_score = maximum cosine similarity to any class exemplar
      prototype_score = cosine similarity to the normalized class-mean embedding
      hybrid_score = exemplar_weight * exemplar_score + prototype_weight * prototype_score

    The default 50/50 weighting is fixed by design for this experiment and is not
    tuned from query/test labels.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: list[str],
        image_paths: list[str],
        exemplar_weight: float = 0.50,
        prototype_weight: float = 0.50,
    ):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
        if len(embeddings) == 0:
            raise ValueError("defect bank cannot be empty")
        if len(labels) != len(embeddings) or len(image_paths) != len(embeddings):
            raise ValueError("embeddings, labels and image_paths must have equal length")

        weight_sum = float(exemplar_weight + prototype_weight)
        if weight_sum <= 0:
            raise ValueError("bank weights must sum to a positive value")
        self.exemplar_weight = float(exemplar_weight) / weight_sum
        self.prototype_weight = float(prototype_weight) / weight_sum

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-12, None)
        self.labels = list(labels)
        self.image_paths = list(image_paths)
        self.classes = sorted(set(self.labels))

        prototype_rows = []
        for class_name in self.classes:
            indices = [i for i, label in enumerate(self.labels) if label == class_name]
            proto = self.embeddings[indices].mean(axis=0)
            proto = proto / max(float(np.linalg.norm(proto)), 1e-12)
            prototype_rows.append(proto.astype(np.float32))
        self.prototypes = np.stack(prototype_rows, axis=0)

    def save(self, bank_dir: str | Path) -> Path:
        bank_dir = Path(bank_dir)
        bank_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bank_dir / "embeddings.npz",
            embeddings=self.embeddings,
            prototypes=self.prototypes,
        )
        metadata = {
            "format_version": 1,
            "embedding_dim": int(self.embeddings.shape[1]),
            "num_exemplars": int(self.embeddings.shape[0]),
            "classes": self.classes,
            "labels": self.labels,
            "image_paths": self.image_paths,
            "class_score": "fixed_exemplar_plus_prototype",
            "exemplar_weight": self.exemplar_weight,
            "prototype_weight": self.prototype_weight,
        }
        with open(bank_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return bank_dir

    @classmethod
    def load(cls, bank_dir: str | Path) -> "HybridDefectBank":
        bank_dir = Path(bank_dir)
        arrays = np.load(bank_dir / "embeddings.npz")
        with open(bank_dir / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(
            arrays["embeddings"].astype(np.float32),
            metadata["labels"],
            metadata["image_paths"],
            exemplar_weight=float(metadata.get("exemplar_weight", 0.50)),
            prototype_weight=float(metadata.get("prototype_weight", 0.50)),
        )

    def score_embedding(self, embedding: np.ndarray) -> dict:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: {query.shape[0]} vs {self.embeddings.shape[1]}"
            )

        exemplar_sims = self.embeddings @ query
        prototype_sims = self.prototypes @ query

        exemplar_scores: dict[str, float] = {}
        prototype_scores: dict[str, float] = {}
        hybrid_scores: dict[str, float] = {}
        class_best_index: dict[str, int] = {}

        for class_index, class_name in enumerate(self.classes):
            indices = [i for i, label in enumerate(self.labels) if label == class_name]
            local = exemplar_sims[indices]
            best_local = int(np.argmax(local))
            best_index = indices[best_local]
            exemplar_score = float(exemplar_sims[best_index])
            prototype_score = float(prototype_sims[class_index])
            hybrid_score = (
                self.exemplar_weight * exemplar_score
                + self.prototype_weight * prototype_score
            )
            exemplar_scores[class_name] = exemplar_score
            prototype_scores[class_name] = prototype_score
            hybrid_scores[class_name] = float(hybrid_score)
            class_best_index[class_name] = best_index

        return {
            "exemplar_scores": exemplar_scores,
            "prototype_scores": prototype_scores,
            "hybrid_scores": hybrid_scores,
            "class_best_index": class_best_index,
        }

    @staticmethod
    def _rank(scores: dict[str, float]) -> dict:
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top1_class, top1_score = ranked[0]
        if len(ranked) > 1:
            top2_class, top2_score = ranked[1]
            margin = float(top1_score - top2_score)
        else:
            top2_class, top2_score, margin = None, float("-inf"), float("inf")
        return {
            "predicted_class": top1_class,
            "top1_similarity": float(top1_score),
            "top2_class": top2_class,
            "top2_similarity": float(top2_score),
            "margin": margin,
            "class_scores": scores,
        }

    def predict_embedding(self, embedding: np.ndarray, mode: str = "hybrid") -> dict:
        components = self.score_embedding(embedding)
        mode = str(mode).strip().lower()
        key = {
            "exemplar": "exemplar_scores",
            "prototype": "prototype_scores",
            "hybrid": "hybrid_scores",
        }.get(mode)
        if key is None:
            raise ValueError("mode must be 'exemplar', 'prototype', or 'hybrid'")
        ranked = self._rank(components[key])
        best_index = components["class_best_index"][ranked["predicted_class"]]
        return {
            **ranked,
            "mode": mode,
            "nearest_exemplar": self.image_paths[best_index],
            "exemplar_scores": components["exemplar_scores"],
            "prototype_scores": components["prototype_scores"],
            "hybrid_scores": components["hybrid_scores"],
        }
