from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_clean_path(value: str | Path) -> Path:
    """Resolve CLI paths consistently relative to industrial_anomaly/."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build a KNOWN defect exemplar bank from PatchCore-located ROIs. "
            "No classifier training or DINOv2 fine-tuning is performed."
        )
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. phone")
    p.add_argument(
        "--defects-dir",
        default=None,
        help=(
            "Folder containing defect-class subfolders. Relative paths are resolved from "
            "industrial_anomaly/. Default: products/<product>/defects"
        ),
    )
    p.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help=(
            "Optional explicit class-folder names, e.g. --classes shao1 shao2 shao3. "
            "Use this when defects-dir also contains good/test folders."
        ),
    )
    p.add_argument(
        "--patchcore-model-dir",
        default=None,
        help=(
            "Relative paths are resolved from industrial_anomaly/. "
            "Default: products/<product>/models/patchcore"
        ),
    )
    p.add_argument(
        "--bank-dir",
        default=None,
        help=(
            "Relative paths are resolved from industrial_anomaly/. "
            "Default: products/<product>/models/defect_bank"
        ),
    )
    p.add_argument(
        "--shots",
        type=int,
        default=10,
        help="Samples/class. Use 0 for all available images.",
    )
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
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
        seen = set()
        names = []
        for name in requested:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                names.append(name)

        class_dirs = []
        missing = []
        for name in names:
            folder = defects_dir / name
            if not folder.is_dir():
                missing.append(name)
            else:
                class_dirs.append(folder)
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
    defects_dir = (
        resolve_clean_path(args.defects_dir)
        if args.defects_dir
        else product_dir / "defects"
    )
    patchcore_model_dir = (
        resolve_clean_path(args.patchcore_model_dir)
        if args.patchcore_model_dir
        else product_dir / "models" / "patchcore"
    )
    bank_dir = (
        resolve_clean_path(args.bank_dir)
        if args.bank_dir
        else product_dir / "models" / "defect_bank"
    )

    if not defects_dir.exists():
        raise FileNotFoundError(f"Defect folder not found: {defects_dir.resolve()}")
    if not patchcore_model_dir.exists():
        raise FileNotFoundError(
            f"PatchCore model not found: {patchcore_model_dir.resolve()}\n"
            "Run train_patchcore.py first."
        )

    class_dirs = select_class_dirs(defects_dir, args.classes)
    if not class_dirs:
        raise RuntimeError(
            f"No defect class directories found in {defects_dir.resolve()}\n"
            "Use --classes <class1> <class2> ... when the dataset root also contains good/test."
        )

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
    print(f"clean root:      {CLEAN_ROOT}")
    print(f"repo root:       {REPO_ROOT}")
    print(f"product:         {args.product}")
    print(f"defects dir:     {defects_dir.resolve()}")
    print(f"classes:         {[d.name for d in class_dirs]}")
    print(f"PatchCore:       {patchcore_model_dir.resolve()}")
    print(f"bank dir:        {bank_dir.resolve()}")
    print(f"shots/class:     {'ALL' if args.shots == 0 else args.shots}")
    print("feature fusion:  50% DINOv2 CLS + 50% DINOv2 Patch Center")
    print("=============================================")

    for class_dir in class_dirs:
        images = image_files(class_dir)
        if not images:
            raise RuntimeError(f"Class {class_dir.name!r} has no images")
        if args.shots > 0 and len(images) < args.shots:
            raise RuntimeError(
                f"Class {class_dir.name!r} has {len(images)} images, fewer than shots={args.shots}. "
                "Use a smaller --shots or --shots 0."
            )
        chosen = images if args.shots == 0 else images[: args.shots]
        class_counts[class_dir.name] = len(chosen)
        print(f"\n[{class_dir.name}] support={len(chosen)}")

        for image_path in chosen:
            roi_result = pipeline.extract_roi(image_path)
            cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
            center_embedding = pipeline.embed_roi(
                roi_result["roi"],
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )

            cls_embeddings.append(cls_embedding)
            center_embeddings.append(center_embedding)
            labels.append(class_dir.name)
            source_paths.append(str(image_path.resolve()))

            roi_path = bank_dir / "support_rois" / class_dir.name / image_path.name
            overlay_path = bank_dir / "support_overlays" / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)
            pipeline.save_bbox_overlay(
                roi_result["display_image"], roi_result["bbox"], overlay_path
            )

            print(
                f"  + {image_path.name} score={roi_result['anomaly_score']:.3f} "
                f"bbox={roi_result['bbox']} original_bbox={roi_result['original_bbox']}"
            )

    cls_bank = DefectExemplarBank(np.stack(cls_embeddings), labels, source_paths)
    center_bank = DefectExemplarBank(np.stack(center_embeddings), labels, source_paths)
    cls_bank.save(bank_dir / "cls")
    center_bank.save(bank_dir / "center")

    bank_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "format_version": 2,
        "product": args.product,
        "classes": cls_bank.classes,
        "class_counts": class_counts,
        "num_exemplars": len(labels),
        "features": ["dinov2_cls", "dinov2_patch_center"],
        "center_fraction": float(args.center_fraction),
        "fusion_cls_weight": 0.50,
        "fusion_center_weight": 0.50,
        "class_score": "0.5*max_cosine_cls + 0.5*max_cosine_patch_center",
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
