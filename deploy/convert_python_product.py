from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import struct
from pathlib import Path

import faiss
import numpy as np


MAGIC = b"F32M"


def log(message: str) -> None:
    print(message, flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_binary_matrix(path: Path, matrix: np.ndarray) -> None:
    matrix = np.asarray(matrix, dtype=np.float32, order="C")
    if matrix.ndim != 2:
        raise ValueError(f"matrix must be 2-D, got {matrix.shape}")
    rows, cols = matrix.shape
    if rows <= 0 or cols <= 0:
        raise ValueError(f"matrix must be non-empty, got {matrix.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<ii", rows, cols))
        f.write(matrix.astype("<f4", copy=False).tobytes(order="C"))


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def reconstruct_faiss_memory(index_path: Path) -> tuple[np.ndarray, dict]:
    if not index_path.exists():
        raise FileNotFoundError(index_path)

    log(f"[PatchCore] Reading FAISS index: {index_path}")
    index = faiss.read_index(str(index_path))
    rows = int(index.ntotal)
    cols = int(index.d)
    if rows <= 0:
        raise RuntimeError("FAISS index is empty")

    metric_type = getattr(index, "metric_type", faiss.METRIC_L2)
    if metric_type != faiss.METRIC_L2:
        raise RuntimeError(
            f"Expected L2 FAISS index, metric_type={metric_type}"
        )

    memory = None
    try:
        reconstructed = index.reconstruct_n(0, rows)
        if reconstructed is not None:
            memory = np.asarray(reconstructed, dtype=np.float32)
    except TypeError:
        pass

    if memory is None:
        memory = np.empty((rows, cols), dtype=np.float32)
        try:
            index.reconstruct_n(0, rows, memory)
        except TypeError:
            for i in range(rows):
                memory[i] = np.asarray(index.reconstruct(i), dtype=np.float32)

    memory = np.asarray(memory, dtype=np.float32).reshape(rows, cols)

    # Verify that the reconstructed vectors reproduce the original FAISS L2
    # nearest-neighbour distances.  Queries are slightly perturbed memory rows,
    # so this is not the trivial self-distance == 0 case.
    rng = np.random.default_rng(20260815)
    sample_count = min(8, rows)
    selected = rng.choice(rows, size=sample_count, replace=False)
    queries = memory[selected].copy()
    scale = float(np.std(memory))
    if not np.isfinite(scale) or scale <= 1e-12:
        scale = 1.0
    queries += rng.normal(0.0, scale * 1e-4, size=queries.shape).astype(np.float32)

    faiss_dist, faiss_idx = index.search(queries, 1)
    # Do the comparison in chunks to avoid allocating Q x M x D.
    numpy_dist = np.full(sample_count, np.inf, dtype=np.float64)
    numpy_idx = np.full(sample_count, -1, dtype=np.int64)
    chunk = 2048
    for start in range(0, rows, chunk):
        block = memory[start : start + chunk]
        for qi, query in enumerate(queries):
            diff = block - query
            dist = np.einsum("ij,ij->i", diff, diff, dtype=np.float64)
            local = int(np.argmin(dist))
            value = float(dist[local])
            if value < numpy_dist[qi]:
                numpy_dist[qi] = value
                numpy_idx[qi] = start + local

    max_abs = float(np.max(np.abs(faiss_dist[:, 0].astype(np.float64) - numpy_dist)))
    index_match = bool(np.all(faiss_idx[:, 0].astype(np.int64) == numpy_idx))
    tolerance = max(1e-5, float(np.max(np.abs(faiss_dist[:, 0]))) * 1e-4)
    if max_abs > tolerance or not index_match:
        raise RuntimeError(
            "Reconstructed FAISS memory failed verification: "
            f"max_abs={max_abs:.6g}, tolerance={tolerance:.6g}, "
            f"index_match={index_match}"
        )

    info = {
        "index_type": type(index).__name__,
        "rows": rows,
        "cols": cols,
        "metric": "L2",
        "verification_queries": sample_count,
        "verification_max_abs": max_abs,
        "verification_index_match": index_match,
    }
    log(
        f"[PatchCore] Reconstructed exact memory: {rows} x {cols}; "
        f"FAISS verification max_abs={max_abs:.6g}"
    )
    return memory, info


def load_optional_saved_features(model_dir: Path) -> np.ndarray | None:
    path = model_dir / "nnscorer_features.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        value = pickle.load(f)
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"Unexpected nnscorer_features.pkl shape: {matrix.shape}")
    return matrix


def load_defect_bank(bank_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], list[str], dict]:
    cls_dir = bank_dir / "cls"
    center_dir = bank_dir / "center"

    cls_npz = cls_dir / "embeddings.npz"
    center_npz = center_dir / "embeddings.npz"
    cls_meta_path = cls_dir / "metadata.json"
    center_meta_path = center_dir / "metadata.json"

    cls = np.load(cls_npz)["embeddings"].astype(np.float32, copy=False)
    center = np.load(center_npz)["embeddings"].astype(np.float32, copy=False)
    cls_meta = read_json(cls_meta_path)
    center_meta = read_json(center_meta_path)

    labels = list(cls_meta["labels"])
    center_labels = list(center_meta["labels"])
    if labels != center_labels:
        raise RuntimeError("CLS and Center bank labels differ")
    if cls.shape != center.shape:
        raise RuntimeError(f"CLS/Center shapes differ: {cls.shape} vs {center.shape}")
    if cls.ndim != 2 or cls.shape[1] != 384:
        raise RuntimeError(f"Expected DINO bank [N,384], got {cls.shape}")
    if len(labels) != cls.shape[0]:
        raise RuntimeError("Defect labels do not match embedding rows")

    bank_cfg_path = bank_dir / "bank_config.json"
    bank_cfg = read_json(bank_cfg_path) if bank_cfg_path.exists() else {}
    classes = list(bank_cfg.get("classes") or cls_meta.get("classes") or sorted(set(labels)))

    cls_norm_error = float(np.max(np.abs(np.linalg.norm(cls, axis=1) - 1.0)))
    center_norm_error = float(np.max(np.abs(np.linalg.norm(center, axis=1) - 1.0)))
    log(
        f"[DINO] CLS={cls.shape}, Center={center.shape}, classes={classes}; "
        f"norm_error cls={cls_norm_error:.3g} center={center_norm_error:.3g}"
    )

    return cls, center, labels, classes, {
        "cls_norm_max_error": cls_norm_error,
        "center_norm_max_error": center_norm_error,
        "bank_config": bank_cfg,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the ORIGINAL Python PatchCore FAISS memory and DINOv2 "
            "exemplar banks into the C# F32M product format. No C# rebuilding, "
            "resampling or re-embedding is performed."
        )
    )
    parser.add_argument("--product", required=True)
    parser.add_argument("--patchcore-model-dir", required=True)
    parser.add_argument("--defect-bank-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.78)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument("--copy-support-rois", action="store_true")
    args = parser.parse_args()

    patchcore_dir = Path(args.patchcore_model_dir).expanduser().resolve()
    defect_bank_dir = Path(args.defect_bank_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    params = read_json(patchcore_dir / "inspection_config.json") if (patchcore_dir / "inspection_config.json").exists() else {}
    tile_fraction = float(params.get("tile_fraction", 0.75))
    tile_overlap = float(params.get("tile_overlap", 0.25))

    memory, faiss_info = reconstruct_faiss_memory(
        patchcore_dir / "nnscorer_search_index.faiss"
    )
    if memory.shape[1] != 1024:
        raise RuntimeError(f"Expected PatchCore memory dim 1024, got {memory.shape}")

    saved_features = load_optional_saved_features(patchcore_dir)
    feature_source = "faiss_reconstruct"
    if saved_features is not None:
        if saved_features.shape != memory.shape:
            raise RuntimeError(
                f"Saved scorer features shape {saved_features.shape} does not match "
                f"FAISS memory {memory.shape}"
            )
        max_abs = float(np.max(np.abs(saved_features - memory)))
        if max_abs > 1e-6:
            raise RuntimeError(
                f"nnscorer_features.pkl differs from FAISS reconstructed memory: {max_abs:.6g}"
            )
        memory = saved_features
        feature_source = "nnscorer_features.pkl_verified_against_faiss"
        log(f"[PatchCore] Saved feature matrix matches FAISS exactly: max_abs={max_abs:.6g}")

    cls, center, labels, classes, defect_info = load_defect_bank(defect_bank_dir)
    bank_cfg = defect_info["bank_config"]
    cls_weight = float(bank_cfg.get("fusion_cls_weight", 0.50))
    center_weight = float(bank_cfg.get("fusion_center_weight", 0.50))

    memory_out = output_dir / "patchcore_memory.bin"
    cls_out = output_dir / "defect_cls.bin"
    center_out = output_dir / "defect_center.bin"
    write_binary_matrix(memory_out, memory)
    write_binary_matrix(cls_out, cls)
    write_binary_matrix(center_out, center)

    manifest = {
        "FormatVersion": 2,
        "ProductName": args.product,
        "ProductModelSource": "python_export",
        "PatchCoreMemoryFile": memory_out.name,
        "DefectClsFile": cls_out.name,
        "DefectCenterFile": center_out.name,
        "DefectLabels": labels,
        "Classes": classes,
        "TileFraction": tile_fraction,
        "TileOverlap": tile_overlap,
        "CoresetRatio": 0.0,
        "PatchCoreMemoryRows": int(memory.shape[0]),
        "PatchCoreMemoryStrategy": "python_faiss_memory_exact",
        "BboxRelativeThreshold": float(args.bbox_relative_threshold),
        "RoiMargin": float(args.roi_margin),
        "ClsWeight": cls_weight,
        "CenterWeight": center_weight,
    }
    (output_dir / "product_model.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.copy_support_rois:
        source_rois = defect_bank_dir / "support_rois"
        if source_rois.exists():
            target_rois = output_dir / "support_rois"
            if target_rois.exists():
                shutil.rmtree(target_rois)
            shutil.copytree(source_rois, target_rois)

    report = {
        "format_version": 1,
        "product": args.product,
        "source": {
            "patchcore_model_dir": str(patchcore_dir),
            "defect_bank_dir": str(defect_bank_dir),
            "patchcore_index_sha256": sha256_file(patchcore_dir / "nnscorer_search_index.faiss"),
            "cls_npz_sha256": sha256_file(defect_bank_dir / "cls" / "embeddings.npz"),
            "center_npz_sha256": sha256_file(defect_bank_dir / "center" / "embeddings.npz"),
        },
        "patchcore": {
            **faiss_info,
            "feature_source": feature_source,
        },
        "dinov2": {
            "rows": int(cls.shape[0]),
            "cols": int(cls.shape[1]),
            "classes": classes,
            "cls_norm_max_error": defect_info["cls_norm_max_error"],
            "center_norm_max_error": defect_info["center_norm_max_error"],
        },
        "output": {
            "patchcore_memory_sha256": sha256_file(memory_out),
            "defect_cls_sha256": sha256_file(cls_out),
            "defect_center_sha256": sha256_file(center_out),
        },
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log("============================================")
    log("Python product model conversion finished.")
    log(f"output: {output_dir}")
    log(f"PatchCore memory: {memory.shape[0]} x {memory.shape[1]}")
    log(f"DINO exemplars:   {cls.shape[0]} x {cls.shape[1]}")
    log(f"classes:          {classes}")
    log("strategy:         python_faiss_memory_exact")
    log("C# rebuilding/reservoir sampling: DISABLED for this product")


if __name__ == "__main__":
    main()
