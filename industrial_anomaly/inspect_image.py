from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.anomaly.postprocessing import normalize_anomaly_map
from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


def resolve_clean_path(value: str | Path) -> Path:
    """Resolve CLI paths consistently relative to industrial_anomaly/."""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Inspect one image: PatchCore anomaly localization -> DINOv2 known-defect classification."
        )
    )
    p.add_argument(
        "image",
        help=(
            "Input image path. Relative paths are resolved from industrial_anomaly/."
        ),
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. bottle")
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
        "--output-dir",
        default=None,
        help=(
            "Relative paths are resolved from industrial_anomaly/. "
            "Default: outputs/<product>/<image_stem>"
        ),
    )
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument(
        "--anomaly-threshold",
        type=float,
        default=None,
        help=(
            "Optional calibrated PatchCore PASS/NG threshold. If omitted, the script reports "
            "anomaly score and known-defect candidate without making a PASS/NG decision."
        ),
    )
    return p.parse_args()


def rank_scores(scores: dict[str, float]) -> dict:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
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
    }


def save_anomaly_map(anomaly_map: np.ndarray, output_path: Path) -> None:
    norm = normalize_anomaly_map(np.asarray(anomaly_map, dtype=np.float32))
    image = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def main():
    args = parse_args()
    image_path = resolve_clean_path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    product_dir = CLEAN_ROOT / "products" / args.product
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
    output_dir = (
        resolve_clean_path(args.output_dir)
        if args.output_dir
        else CLEAN_ROOT / "outputs" / args.product / image_path.stem
    )

    if not patchcore_model_dir.exists():
        raise FileNotFoundError(f"PatchCore model not found: {patchcore_model_dir.resolve()}")
    if not (bank_dir / "cls").exists() or not (bank_dir / "center").exists():
        raise FileNotFoundError(
            f"Defect bank not found: {bank_dir.resolve()}\n"
            "Run build_defect_bank.py first."
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

    cls_bank = DefectExemplarBank.load(bank_dir / "cls")
    center_bank = DefectExemplarBank.load(bank_dir / "center")
    if cls_bank.classes != center_bank.classes:
        raise RuntimeError("CLS and Patch Center banks contain different classes")

    roi_result = pipeline.extract_roi(image_path)
    cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
    center_embedding = pipeline.embed_roi(
        roi_result["roi"],
        feature_mode="patch_center",
        center_fraction=args.center_fraction,
    )

    cls_result = cls_bank.predict_embedding(cls_embedding)
    center_result = center_bank.predict_embedding(center_embedding)
    fused_scores = {
        class_name: 0.50 * cls_result["class_scores"][class_name]
        + 0.50 * center_result["class_scores"][class_name]
        for class_name in cls_bank.classes
    }
    fused = rank_scores(fused_scores)

    if args.anomaly_threshold is None:
        anomaly_decision = "UNCALIBRATED"
        final_result = f"KNOWN_DEFECT_CANDIDATE: {fused['predicted_class']}"
    elif roi_result["anomaly_score"] < args.anomaly_threshold:
        anomaly_decision = "PASS"
        final_result = "PASS"
    else:
        anomaly_decision = "NG"
        final_result = f"NG: {fused['predicted_class']}"

    output_dir.mkdir(parents=True, exist_ok=True)
    roi_path = output_dir / "roi.png"
    bbox_path = output_dir / "bbox.jpg"
    anomaly_path = output_dir / "anomaly_map.png"
    result_path = output_dir / "result.json"

    roi_result["roi"].save(roi_path)
    pipeline.save_bbox_overlay(roi_result["display_image"], roi_result["bbox"], bbox_path)
    save_anomaly_map(roi_result["anomaly_map"], anomaly_path)

    payload = {
        "product": args.product,
        "image": str(image_path.resolve()),
        "patchcore_anomaly_score": float(roi_result["anomaly_score"]),
        "anomaly_threshold": args.anomaly_threshold,
        "anomaly_decision": anomaly_decision,
        "bbox": list(roi_result["bbox"]),
        "bbox_source": roi_result["bbox_source"],
        "predicted_known_defect": fused["predicted_class"],
        "top1_similarity": fused["top1_similarity"],
        "top2_class": fused["top2_class"],
        "top2_similarity": fused["top2_similarity"],
        "margin": fused["margin"],
        "cls_nearest_exemplar": cls_result["nearest_exemplar"],
        "center_nearest_exemplar": center_result["nearest_exemplar"],
        "final_result": final_result,
        "note": (
            "PASS/NG is only valid when --anomaly-threshold is supplied from a separate "
            "non-leaky calibration set. Known-defect classification uses 50/50 CLS + Patch Center."
        ),
    }
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("========== Industrial Anomaly Inspection ==========")
    print(f"clean root:           {CLEAN_ROOT}")
    print(f"repo root:            {REPO_ROOT}")
    print(f"product:              {args.product}")
    print(f"image:                {image_path.resolve()}")
    print(f"PatchCore model:      {patchcore_model_dir.resolve()}")
    print(f"defect bank:          {bank_dir.resolve()}")
    print(f"PatchCore score:      {roi_result['anomaly_score']:.6f}")
    if args.anomaly_threshold is None:
        print("PASS/NG threshold:    NOT SET (score only)")
    else:
        print(f"PASS/NG threshold:    {args.anomaly_threshold:.6f}")
        print(f"anomaly decision:     {anomaly_decision}")
    print(f"bbox:                 {roi_result['bbox']}")
    print(f"bbox source:          {roi_result['bbox_source']}")
    print(f"known defect:         {fused['predicted_class']}")
    print(f"Top-1 similarity:     {fused['top1_similarity']:.6f}")
    print(f"Top-2 class:          {fused['top2_class']}")
    print(f"margin:               {fused['margin']:.6f}")
    print(f"final:                {final_result}")
    print(f"output dir:           {output_dir.resolve()}")
    print(f"ROI:                  {roi_path.resolve()}")
    print(f"BBox:                 {bbox_path.resolve()}")
    print(f"Anomaly map:          {anomaly_path.resolve()}")
    print(f"JSON:                 {result_path.resolve()}")
    print("===================================================")


if __name__ == "__main__":
    main()
