from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.anomaly.tiled import (
    crop_square_with_margin,
    inspect_tiled_patchcore,
    load_inspection_config,
    save_regions_overlay,
)
from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_clean_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def parse_args():
    p = argparse.ArgumentParser(
        description="Build a known-defect exemplar bank from PatchCore-located ROIs."
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. phone")
    p.add_argument("--defects-dir", default=None)
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Explicit class folders, e.g. --classes shao1 shao2 shao3",
    )
    p.add_argument("--patchcore-model-dir", default=None)
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--bbox-relative-threshold", type=float, default=0.78)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    return p.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def select_class_dirs(defects_dir: Path, requested: list[str] | None) -> list[Path]:
    if requested:
        class_dirs, missing = [], []
        seen = set()
        for name in requested:
            key = name.lower()
            if key in seen:
                continue
            seen.add(key)
            folder = defects_dir / name
            if folder.is_dir():
                class_dirs.append(folder)
            else:
                missing.append(name)
        if missing:
            raise FileNotFoundError(
                f"Requested defect class folders not found under {defects_dir}: {missing}"
            )
        return class_dirs

    ignored = {"good", "test", "train", "normal"}
    return sorted(
        [
            d
            for d in defects_dir.iterdir()
            if d.is_dir()
            and d.name.lower() not in ignored
            and not d.name.startswith("_")
        ],
        key=lambda p: p.name.lower(),
    )


def main():
    args = parse_args()
    if args.shots < 0:
        raise ValueError("--shots must be >= 0")

    product_dir = CLEAN_ROOT / "products" / args.product
    defects_dir = resolve_clean_path(args.defects_dir) if args.defects_dir else product_dir / "defects"
    patchcore_model_dir = (
        resolve_clean_path(args.patchcore_model_dir)
        if args.patchcore_model_dir
        else product_dir / "models" / "patchcore"
    )
    bank_dir = resolve_clean_path(args.bank_dir) if args.bank_dir else product_dir / "models" / "defect_bank"

    if not defects_dir.is_dir():
        raise FileNotFoundError(f"Defect folder not found: {defects_dir}")
    if not patchcore_model_dir.exists():
        raise FileNotFoundError(
            f"PatchCore model not found: {patchcore_model_dir}\nRun train_patchcore.py first."
        )

    class_dirs = select_class_dirs(defects_dir, args.classes)
    if not class_dirs:
        raise RuntimeError("No defect class directories found.")

    inspection_cfg = load_inspection_config(patchcore_model_dir)
    inspection_mode = inspection_cfg.get("mode", "center_crop")
    tile_fraction = float(inspection_cfg.get("tile_fraction", 0.75))
    tile_overlap = float(inspection_cfg.get("tile_overlap", 0.25))

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
        center_fraction=args.center_fraction,
    )
    pipeline.load()

    cls_embeddings: list[np.ndarray] = []
    center_embeddings: list[np.ndarray] = []
    labels: list[str] = []
    source_paths: list[str] = []
    class_counts: dict[str, int] = {}

    print("========== Build Known Defect Bank ==========")
    print(f"product:          {args.product}")
    print(f"defects dir:      {defects_dir}")
    print(f"classes:          {[d.name for d in class_dirs]}")
    print(f"PatchCore:        {patchcore_model_dir}")
    print(f"inspection mode:  {inspection_mode}")
    if inspection_mode == "tiled":
        print(f"tile fraction:    {tile_fraction}")
        print(f"tile overlap:     {tile_overlap}")
    print(f"shots/class:      {'ALL' if args.shots == 0 else args.shots}")
    print("feature fusion:   50% DINOv2 CLS + 50% DINOv2 Patch Center")
    print("=============================================")

    for class_dir in class_dirs:
        images = image_files(class_dir)
        if not images:
            raise RuntimeError(f"Class {class_dir.name!r} has no images")
        if args.shots > 0 and len(images) < args.shots:
            raise RuntimeError(
                f"Class {class_dir.name!r} has {len(images)} images, fewer than shots={args.shots}."
            )
        chosen = images if args.shots == 0 else images[: args.shots]
        class_counts[class_dir.name] = len(chosen)
        print(f"\n[{class_dir.name}] support={len(chosen)}")

        for image_path in chosen:
            if inspection_mode == "tiled":
                tiled = inspect_tiled_patchcore(
                    pipeline.patchcore,
                    image_path,
                    tile_fraction=tile_fraction,
                    overlap=tile_overlap,
                    relative_threshold=args.bbox_relative_threshold,
                )
                if not tiled["regions"]:
                    raise RuntimeError(
                        f"No tiled anomaly region found for support image: {image_path}"
                    )
                primary = tiled["regions"][0]
                original = Image.open(image_path).convert("RGB")
                roi = crop_square_with_margin(
                    original,
                    primary["bbox"],
                    margin=args.roi_margin,
                )
                anomaly_score = tiled["anomaly_score"]
                bbox_for_log = primary["bbox"]
                region_count = len(tiled["regions"])
            else:
                roi_result = pipeline.extract_roi(image_path)
                roi = roi_result["roi"]
                anomaly_score = roi_result["anomaly_score"]
                bbox_for_log = roi_result["original_bbox"]
                region_count = 1

            cls_embedding = pipeline.embed_roi(roi, feature_mode="cls")
            center_embedding = pipeline.embed_roi(
                roi,
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )
            cls_embeddings.append(cls_embedding)
            center_embeddings.append(center_embedding)
            labels.append(class_dir.name)
            source_paths.append(str(image_path.resolve()))

            roi_path = bank_dir / "support_rois" / class_dir.name / image_path.with_suffix(".png").name
            overlay_path = bank_dir / "support_overlays" / class_dir.name / image_path.with_suffix(".jpg").name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi.save(roi_path)

            if inspection_mode == "tiled":
                save_regions_overlay(image_path, tiled["regions"], overlay_path)
            else:
                pipeline.save_full_image_overlay(
                    image_path,
                    roi_result["original_bbox"],
                    overlay_path,
                    label=class_dir.name,
                    anomaly_score=anomaly_score,
                )

            print(
                f"  + {image_path.name} score={anomaly_score:.3f} "
                f"primary_bbox={bbox_for_log} regions={region_count}"
            )

    cls_bank = DefectExemplarBank(np.stack(cls_embeddings), labels, source_paths)
    center_bank = DefectExemplarBank(np.stack(center_embeddings), labels, source_paths)
    cls_bank.save(bank_dir / "cls")
    center_bank.save(bank_dir / "center")

    bank_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 3,
        "product": args.product,
        "classes": cls_bank.classes,
        "class_counts": class_counts,
        "num_exemplars": len(labels),
        "inspection_mode": inspection_mode,
        "tile_fraction": tile_fraction if inspection_mode == "tiled" else None,
        "tile_overlap": tile_overlap if inspection_mode == "tiled" else None,
        "features": ["dinov2_cls", "dinov2_patch_center"],
        "center_fraction": float(args.center_fraction),
        "fusion_cls_weight": 0.50,
        "fusion_center_weight": 0.50,
        "source": "PatchCore-predicted ROI; no GT mask used",
    }
    with open(bank_dir / "bank_config.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print("\nKnown defect bank finished.")
    print(f"classes: {cls_bank.classes}")
    print(f"counts:  {class_counts}")
    print(f"bank:    {bank_dir.resolve()}")


if __name__ == "__main__":
    main()
