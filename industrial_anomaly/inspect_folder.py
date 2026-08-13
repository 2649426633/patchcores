from __future__ import annotations

import argparse
import csv
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


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_clean_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


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


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch inspect a folder: PatchCore localization + known-defect classification."
    )
    p.add_argument(
        "test_dir",
        help="Folder containing test images. Relative paths resolve from industrial_anomaly/.",
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. phone")
    p.add_argument("--patchcore-model-dir", default=None)
    p.add_argument("--bank-dir", default=None)
    p.add_argument(
        "--output-dir",
        default=None,
        help="Default: outputs/<product>/test",
    )
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument(
        "--anomaly-threshold",
        type=float,
        default=None,
        help="Optional independently calibrated PASS/NG threshold.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    test_dir = resolve_clean_path(args.test_dir)
    if not test_dir.is_dir():
        raise NotADirectoryError(f"Test directory not found: {test_dir}")

    images = image_files(test_dir)
    if not images:
        raise RuntimeError(f"No supported images found in: {test_dir}")

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
        else CLEAN_ROOT / "outputs" / args.product / "test"
    )

    if not patchcore_model_dir.exists():
        raise FileNotFoundError(f"PatchCore model not found: {patchcore_model_dir}")
    if not (bank_dir / "cls").exists() or not (bank_dir / "center").exists():
        raise FileNotFoundError(
            f"Defect bank not found: {bank_dir}\nRun build_defect_bank.py first."
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

    marked_dir = output_dir / "marked"
    roi_dir = output_dir / "rois"
    anomaly_dir = output_dir / "anomaly_maps"
    marked_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)
    anomaly_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    print("========== Batch Industrial Inspection ==========")
    print(f"product:          {args.product}")
    print(f"test dir:         {test_dir}")
    print(f"test images:      {len(images)}")
    print(f"PatchCore model:  {patchcore_model_dir}")
    print(f"defect bank:      {bank_dir}")
    print(f"known classes:    {cls_bank.classes}")
    print(f"output dir:       {output_dir}")
    if args.anomaly_threshold is None:
        print("PASS/NG threshold: NOT SET (report score + class candidate)")
    else:
        print(f"PASS/NG threshold: {args.anomaly_threshold:.6f}")
    print("================================================")

    for index, image_path in enumerate(images, start=1):
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
            decision = "UNCALIBRATED"
            final_result = f"KNOWN_DEFECT_CANDIDATE: {fused['predicted_class']}"
            overlay_label = fused["predicted_class"]
        elif roi_result["anomaly_score"] < args.anomaly_threshold:
            decision = "PASS"
            final_result = "PASS"
            overlay_label = "PASS"
        else:
            decision = "NG"
            final_result = f"NG: {fused['predicted_class']}"
            overlay_label = f"NG {fused['predicted_class']}"

        marked_path = marked_dir / f"{image_path.stem}_marked.jpg"
        roi_path = roi_dir / f"{image_path.stem}_roi.png"
        anomaly_path = anomaly_dir / f"{image_path.stem}_anomaly.png"

        roi_result["roi"].save(roi_path)
        save_anomaly_map(roi_result["anomaly_map"], anomaly_path)
        pipeline.save_full_image_overlay(
            image_path=image_path,
            bbox=roi_result["original_bbox"],
            output_path=marked_path,
            label=overlay_label,
            anomaly_score=roi_result["anomaly_score"],
            similarity=fused["top1_similarity"],
        )

        ob = roi_result["original_bbox"]
        cb = roi_result["bbox"]
        row = {
            "image": image_path.name,
            "image_path": str(image_path.resolve()),
            "patchcore_anomaly_score": float(roi_result["anomaly_score"]),
            "anomaly_decision": decision,
            "bbox_source": roi_result["bbox_source"],
            "crop_bbox_x1": int(cb[0]),
            "crop_bbox_y1": int(cb[1]),
            "crop_bbox_x2": int(cb[2]),
            "crop_bbox_y2": int(cb[3]),
            "original_bbox_x1": int(ob[0]),
            "original_bbox_y1": int(ob[1]),
            "original_bbox_x2": int(ob[2]),
            "original_bbox_y2": int(ob[3]),
            "predicted_known_defect": fused["predicted_class"],
            "top1_similarity": float(fused["top1_similarity"]),
            "top2_class": fused["top2_class"],
            "top2_similarity": float(fused["top2_similarity"]),
            "margin": float(fused["margin"]),
            "final_result": final_result,
            "marked_image": str(marked_path.resolve()),
            "roi_image": str(roi_path.resolve()),
            "anomaly_map": str(anomaly_path.resolve()),
        }
        rows.append(row)

        print(
            f"[{index:02d}/{len(images):02d}] {image_path.name} | "
            f"PatchCore={roi_result['anomaly_score']:.4f} | "
            f"bbox={roi_result['original_bbox']} | "
            f"class={fused['predicted_class']} | "
            f"sim={fused['top1_similarity']:.4f} | margin={fused['margin']:.4f}"
        )

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nBatch inspection finished.")
    print(f"CSV:            {csv_path.resolve()}")
    print(f"JSON:           {json_path.resolve()}")
    print(f"Marked images:  {marked_dir.resolve()}")
    print(f"ROIs:           {roi_dir.resolve()}")
    print(f"Anomaly maps:   {anomaly_dir.resolve()}")


if __name__ == "__main__":
    main()
