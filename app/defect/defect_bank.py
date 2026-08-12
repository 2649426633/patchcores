from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DefectExemplarBank:
    """Small frozen-DINOv2 exemplar bank using cosine similarity.

    Embeddings are expected to be L2-normalized. Class score is the maximum
    cosine similarity against all exemplars belonging to that class.
    """

    def __init__(self, embeddings: np.ndarray, labels: list[str], image_paths: list[str]):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError(f"embeddings must be 2-D, got {embeddings.shape}")
        if len(labels) != len(embeddings) or len(image_paths) != len(embeddings):
            raise ValueError("embeddings, labels and image_paths must have equal length")
        if len(embeddings) == 0:
            raise ValueError("defect bank cannot be empty")

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.embeddings = embeddings / np.clip(norms, 1e-12, None)
        self.labels = list(labels)
        self.image_paths = list(image_paths)
        self.classes = sorted(set(self.labels))

    def save(self, bank_dir: str | Path) -> Path:
        bank_dir = Path(bank_dir)
        bank_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(bank_dir / "embeddings.npz", embeddings=self.embeddings)
        metadata = {
            "format_version": 1,
            "embedding_dim": int(self.embeddings.shape[1]),
            "num_exemplars": int(self.embeddings.shape[0]),
            "classes": self.classes,
            "labels": self.labels,
            "image_paths": self.image_paths,
            "class_score": "max_cosine_over_exemplars",
        }
        with open(bank_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        return bank_dir

    @classmethod
    def load(cls, bank_dir: str | Path) -> "DefectExemplarBank":
        bank_dir = Path(bank_dir)
        embeddings_file = bank_dir / "embeddings.npz"
        metadata_file = bank_dir / "metadata.json"
        if not embeddings_file.exists() or not metadata_file.exists():
            raise FileNotFoundError(f"Incomplete defect bank: {bank_dir}")

        embeddings = np.load(embeddings_file)["embeddings"].astype(np.float32)
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return cls(embeddings, metadata["labels"], metadata["image_paths"])

    def predict_embedding(self, embedding: np.ndarray) -> dict:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-12)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError(
                f"embedding dimension mismatch: {query.shape[0]} vs {self.embeddings.shape[1]}"
            )

        similarities = self.embeddings @ query
        class_scores = {}
        class_best_index = {}
        for class_name in self.classes:
            indices = [i for i, label in enumerate(self.labels) if label == class_name]
            local = similarities[indices]
            best_local = int(np.argmax(local))
            best_index = indices[best_local]
            class_scores[class_name] = float(similarities[best_index])
            class_best_index[class_name] = best_index

        ranked = sorted(class_scores.items(), key=lambda item: item[1], reverse=True)
        top1_class, top1_score = ranked[0]
        if len(ranked) >= 2:
            top2_class, top2_score = ranked[1]
        else:
            top2_class, top2_score = None, float("-inf")

        exemplar_index = class_best_index[top1_class]
        return {
            "predicted_class": top1_class,
            "top1_similarity": float(top1_score),
            "top2_class": top2_class,
            "top2_similarity": float(top2_score),
            "margin": float(top1_score - top2_score) if np.isfinite(top2_score) else float("inf"),
            "nearest_exemplar": self.image_paths[exemplar_index],
            "class_scores": class_scores,
        }
