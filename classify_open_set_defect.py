from __future__ import annotations

import argparse
from pathlib import Path

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import FusedOpenSetRecognizer, OpenSetCalibration
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Classify one PatchCore-localized anomaly as a known defect class or Unknown "
            "using frozen DINOv2 CLS + center-patch score fusion."
        )
    )
    parser.add_argument("image")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--open-set-dir",
        default="products/screw/defects/open_set_fusion",
    )
    parser.add_argument("--output-dir", default="outputs/screw/open_set_single")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    open_set_dir = Path(args.open_set_dir)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    cls_bank = DefectExemplarBank.load(open_set_dir / "bank_cls")
    center_bank = DefectExemplarBank.load(open_set_dir / "bank_patch_center")
    calibration = OpenSetCalibration.load(open_set_dir / "open_set_calibration.json")
    recognizer = FusedOpenSetRecognizer(cls_bank, center_bank, calibration)

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    roi_result = pipeline.extract_roi(image_path)
    cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
    center_embedding = pipeline.embed_roi(
        roi_result["roi"],
        feature_mode="patch_center",
        center_fraction=calibration.center_fraction,
    )
    result = recognizer.predict_embeddings(cls_embedding, center_embedding)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    roi_path = output_dir / f"{image_path.stem}_roi.png"
    overlay_path = output_dir / f"{image_path.stem}_bbox.jpg"
    roi_result["roi"].save(roi_path)
    pipeline.save_bbox_overlay(roi_result["display_image"], roi_result["bbox"], overlay_path)

    print("========== PatchCore ROI -> DINOv2 Open Set ==========")
    print(f"image: {image_path.resolve()}")
    print(f"PatchCore anomaly score: {roi_result['anomaly_score']:.6f}")
    print(f"bbox: {roi_result['bbox']}")
    print(f"predicted defect: {result['predicted_class']}")
    print(f"nearest known class: {result['nearest_known_class']}")
    print(f"accepted as known: {result['accepted_as_known']}")
    print(f"fused Top-1 similarity: {result['top1_similarity']:.6f}")
    print(f"similarity threshold: {result['similarity_threshold']:.6f}")
    print(f"Top-2 class: {result['top2_class']}")
    print(f"fused margin: {result['margin']:.6f}")
    print(f"margin threshold: {result['margin_threshold']:.6f}")
    print(f"ROI: {roi_path.resolve()}")
    print(f"BBox overlay: {overlay_path.resolve()}")
    print("NOTE: this command assumes PatchCore has already determined that the image is anomalous.")
    print("======================================================")


if __name__ == "__main__":
    main()
