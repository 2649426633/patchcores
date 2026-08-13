from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import calibrate_from_support
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build CLS + Patch-Center DINOv2 banks from PatchCore ROIs and calibrate "
            "Known/Unknown thresholds using support exemplars only."
        )
    )
    parser.add_argument("--samples-dir", default="products/screw/defects/samples")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--output-dir",
        default="products/screw/defects/open_set_fusion",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="0 uses all available images per class; otherwise use the first N images.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument("--center-fraction", type=float, default=0.50)
    parser.add_argument(
        "--support-quantile",
        type=float,
        default=0.10,
        help="Lower support-only quantile used for Unknown thresholds.",
    )
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)
    if args.shots < 0:
        raise ValueError("--shots must be >= 0")

    class_dirs = sorted(
        [p for p in samples_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if not class_dirs:
        raise RuntimeError(f"No defect class folders found in {samples_dir}")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    cls_embeddings = []
    center_embeddings = []
    labels = []
    source_paths = []

    print("\n========== Build open-set support banks ==========")
    for class_dir in class_dirs:
        files = image_files(class_dir)
        if args.shots > 0:
            files = files[: args.shots]
        if len(files) < 2:
            raise RuntimeError(
                f"Class {class_dir.name!r} needs at least 2 support images for Unknown calibration"
            )

        print(f"{class_dir.name}: {len(files)} support images")
        for image_path in files:
            roi_result = pipeline.extract_roi(image_path)
            cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
            center_embedding = pipeline.embed_roi(
                roi_result["roi"],
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )

            roi_path = output_dir / "support_rois" / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            overlay_path = output_dir / "support_overlays" / class_dir.name / image_path.name
            pipeline.save_bbox_overlay(
                roi_result["display_image"], roi_result["bbox"], overlay_path
            )

            cls_embeddings.append(cls_embedding)
            center_embeddings.append(center_embedding)
            labels.append(class_dir.name)
            source_paths.append(str(image_path.resolve()))
            print(
                f"  + {image_path.name} score={roi_result['anomaly_score']:.3f} "
                f"bbox={roi_result['bbox']}"
            )

    cls_embeddings = np.stack(cls_embeddings, axis=0)
    center_embeddings = np.stack(center_embeddings, axis=0)

    cls_bank = DefectExemplarBank(cls_embeddings, labels, source_paths)
    center_bank = DefectExemplarBank(center_embeddings, labels, source_paths)
    cls_bank.save(output_dir / "bank_cls")
    center_bank.save(output_dir / "bank_patch_center")

    calibration, diagnostics = calibrate_from_support(
        cls_embeddings,
        center_embeddings,
        labels,
        support_quantile=args.support_quantile,
        cls_weight=0.50,
        center_weight=0.50,
        center_fraction=args.center_fraction,
    )
    calibration_path = calibration.save(output_dir / "open_set_calibration.json")

    diagnostic_csv = output_dir / "support_calibration_diagnostics.csv"
    with open(diagnostic_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
        writer.writeheader()
        writer.writerows(diagnostics)

    loo_accuracy = sum(int(r["loo_correct"]) for r in diagnostics) / len(diagnostics)

    print("\n========== Open-set bank ready ==========")
    print(f"classes: {cls_bank.classes}")
    print(f"support exemplars: {len(labels)}")
    print(f"support LOO classification accuracy: {loo_accuracy:.2%}")
    print(f"similarity threshold: {calibration.similarity_threshold:.6f}")
    print(f"margin threshold: {calibration.margin_threshold:.6f}")
    print(f"calibration source: support leave-one-out only")
    print(f"output: {output_dir.resolve()}")
    print(f"calibration: {calibration_path.resolve()}")
    print("=========================================")


if __name__ == "__main__":
    main()
