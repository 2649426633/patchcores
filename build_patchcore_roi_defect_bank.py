from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a DINOv2 defect bank from PatchCore-predicted anomaly ROIs."
    )
    parser.add_argument("--samples-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--bank-dir",
        default="products/screw/defects/bank_patchcore_roi_3shot",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=3,
        help="Number of samples used per class. Use 0 to include every available sample in each class.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def build_bank(args) -> DefectExemplarBank:
    samples_dir = Path(args.samples_dir)
    bank_dir = Path(args.bank_dir)

    if args.shots < 0:
        raise ValueError("--shots must be >= 0; use 0 for all available samples")
    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)

    class_dirs = sorted(
        [p for p in samples_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if not class_dirs:
        raise RuntimeError(f"No defect class directories found in {samples_dir}")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    embeddings = []
    labels = []
    source_paths = []
    class_counts = {}

    print("\n========== PatchCore ROI few-shot support ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if not images:
            raise RuntimeError(f"Class {class_dir.name} has no supported images")
        if args.shots > 0 and len(images) < args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images, fewer than shots={args.shots}"
            )

        chosen = images if args.shots == 0 else images[: args.shots]
        class_counts[class_dir.name] = len(chosen)
        print(f"{class_dir.name}: {len(chosen)} support images")

        for image_path in chosen:
            roi_result = pipeline.extract_roi(image_path)
            embedding = pipeline.embed_roi(roi_result["roi"])

            roi_path = bank_dir / "support_rois" / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            overlay_path = bank_dir / "support_overlays" / class_dir.name / image_path.name
            pipeline.save_bbox_overlay(
                roi_result["display_image"], roi_result["bbox"], overlay_path
            )

            embeddings.append(embedding)
            labels.append(class_dir.name)
            source_paths.append(str(image_path.resolve()))

            print(
                f"  + {image_path.name}  score={roi_result['anomaly_score']:.3f} "
                f"bbox={roi_result['bbox']} source={roi_result['bbox_source']}"
            )

    bank = DefectExemplarBank(
        np.stack(embeddings, axis=0),
        labels,
        source_paths,
    )
    bank.save(bank_dir)

    print("\n========== PatchCore ROI defect bank built ==========")
    print(f"classes: {bank.classes}")
    print(f"samples/class: {class_counts}")
    print(f"num exemplars: {len(bank.labels)}")
    print(f"embedding shape: {bank.embeddings.shape}")
    print(f"bank dir: {bank_dir.resolve()}")
    print("=====================================================")
    return bank


def main():
    args = parse_args()
    build_bank(args)


if __name__ == "__main__":
    main()
