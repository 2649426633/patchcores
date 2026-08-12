from __future__ import annotations

import argparse
from pathlib import Path

from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Classify one anomaly image with PatchCore ROI + frozen DINOv2 bank."
    )
    parser.add_argument("image")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--bank-dir",
        default="products/screw/defects/bank_patchcore_roi_3shot",
    )
    parser.add_argument("--output-dir", default="outputs/screw/patchcore_dinov2_single")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=args.bank_dir,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()
    result = pipeline.classify(image_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_path = output_dir / f"{image_path.stem}_roi.png"
    overlay_path = output_dir / f"{image_path.stem}_bbox.jpg"
    result["roi"].save(roi_path)
    pipeline.save_bbox_overlay(result["display_image"], result["bbox"], overlay_path)

    print("========== PatchCore ROI -> DINOv2 ==========")
    print(f"image: {image_path.resolve()}")
    print(f"PatchCore anomaly score: {result['anomaly_score']:.6f}")
    print(f"bbox: {result['bbox']}")
    print(f"bbox source: {result['bbox_source']}")
    print(f"predicted defect: {result['predicted_class']}")
    print(f"Top-1 similarity: {result['top1_similarity']:.6f}")
    print(f"Top-2 class: {result['top2_class']}")
    print(f"Top-2 similarity: {result['top2_similarity']:.6f}")
    print(f"margin: {result['margin']:.6f}")
    print(f"nearest exemplar: {result['nearest_exemplar']}")
    print(f"ROI: {roi_path.resolve()}")
    print(f"BBox overlay: {overlay_path.resolve()}")
    print("============================================")


if __name__ == "__main__":
    main()
