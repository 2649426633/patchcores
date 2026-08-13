from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.anomaly.postprocessing import normalize_anomaly_map
from app.anomaly.tiled import (
    crop_square_with_margin,
    inspect_tiled_patchcore,
    load_inspection_config,
    save_tiled_heatmap_overlay,
)
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


def classify_roi(pipeline, cls_bank, center_bank, roi, center_fraction: float) -> dict:
    cls_embedding = pipeline.embed_roi(roi, feature_mode="cls")
    center_embedding = pipeline.embed_roi(
        roi,
        feature_mode="patch_center",
        center_fraction=center_fraction,
    )
    cls_result = cls_bank.predict_embedding(cls_embedding)
    center_result = center_bank.predict_embedding(center_embedding)
    fused_scores = {
        class_name: 0.50 * cls_result["class_scores"][class_name]
        + 0.50 * center_result["class_scores"][class_name]
        for class_name in cls_bank.classes
    }
    return rank_scores(fused_scores)


def save_anomaly_map(anomaly_map: np.ndarray, output_path: Path) -> None:
    norm = normalize_anomaly_map(np.asarray(anomaly_map, dtype=np.float32))
    image = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)


def parse_args():
    p = argparse.ArgumentParser(
        description="Batch inspect a folder: PatchCore localization + known-defect classification."
    )
    p.add_argument("test_dir")
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. phone")
    p.add_argument("--patchcore-model-dir", default=None)
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument(
        "--bbox-relative-threshold",
        type=float,
        default=0.70,
        help="Localization-only threshold inside each PatchCore tile.",
    )
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument("--max-regions", type=int, default=6)
    p.add_argument("--anomaly-threshold", type=float, default=None)
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

    cls_bank = DefectExemplarBank.load(bank_dir / "cls")
    center_bank = DefectExemplarBank.load(bank_dir / "center")
    if cls_bank.classes != center_bank.classes:
        raise RuntimeError("CLS and Patch Center banks contain different classes")

    marked_dir = output_dir / "marked"
    roi_dir = output_dir / "rois"
    anomaly_dir = output_dir / "anomaly_maps"
    full_heatmap_dir = output_dir / "full_heatmaps"
    for folder in (marked_dir, roi_dir, anomaly_dir, full_heatmap_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []

    print("========== Batch Industrial Inspection ==========")
    print(f"product:          {args.product}")
    print(f"test dir:         {test_dir}")
    print(f"test images:      {len(images)}")
    print(f"PatchCore model:  {patchcore_model_dir}")
    print(f"inspection mode:  {inspection_mode}")
    if inspection_mode == "tiled":
        print(f"tile fraction:    {tile_fraction}")
        print(f"tile overlap:     {tile_overlap}")
    print(f"defect bank:      {bank_dir}")
    print(f"known classes:    {cls_bank.classes}")
    print(f"output dir:       {output_dir}")
    print("marked output:    FINAL RESULT ONLY")
    print("================================================")

    for index, image_path in enumerate(images, start=1):
        if inspection_mode == "tiled":
            tiled = inspect_tiled_patchcore(
                pipeline.patchcore,
                image_path,
                tile_fraction=tile_fraction,
                overlap=tile_overlap,
                relative_threshold=args.bbox_relative_threshold,
                max_regions=args.max_regions,
            )
            regions = tiled["regions"]
            if not regions:
                print(f"[{index:02d}/{len(images):02d}] {image_path.name} | NO REGION")
                continue

            original = Image.open(image_path).convert("RGB")
            region_results = []
            for region_index, region in enumerate(regions, start=1):
                roi = crop_square_with_margin(
                    original,
                    region["bbox"],
                    margin=args.roi_margin,
                )
                cls = classify_roi(
                    pipeline,
                    cls_bank,
                    center_bank,
                    roi,
                    args.center_fraction,
                )
                result = {**region, **cls, "region_index": region_index}
                region_results.append(result)
                roi.save(roi_dir / f"{image_path.stem}_R{region_index}_roi.png")

            primary = region_results[0]
            anomaly_score = tiled["anomaly_score"]
            predicted = primary["predicted_class"]
            top1_similarity = primary["top1_similarity"]
            margin = primary["margin"]
            primary_bbox = primary["bbox"]

            heatmap_path = full_heatmap_dir / f"{image_path.stem}_heatmap.jpg"
            save_tiled_heatmap_overlay(
                image_path,
                tiled["tile_results"],
                heatmap_path,
            )
            anomaly_path = heatmap_path
            bbox_source = "tiled_multi_region"
            all_regions_json = json.dumps(
                [
                    {
                        "region": r["region_index"],
                        "bbox": list(r["bbox"]),
                        "patchcore_rank": r["rank_score"],
                        "tile_score": r["tile_score"],
                        "class": r["predicted_class"],
                        "similarity": r["top1_similarity"],
                        "margin": r["margin"],
                    }
                    for r in region_results
                ],
                ensure_ascii=False,
            )
            num_regions = len(region_results)
        else:
            roi_result = pipeline.extract_roi(image_path)
            cls = classify_roi(
                pipeline,
                cls_bank,
                center_bank,
                roi_result["roi"],
                args.center_fraction,
            )
            anomaly_score = roi_result["anomaly_score"]
            predicted = cls["predicted_class"]
            top1_similarity = cls["top1_similarity"]
            margin = cls["margin"]
            primary_bbox = roi_result["original_bbox"]
            bbox_source = roi_result["bbox_source"]
            num_regions = 1
            all_regions_json = json.dumps(
                [{"region": 1, "bbox": list(primary_bbox), "class": predicted}],
                ensure_ascii=False,
            )

            roi_result["roi"].save(roi_dir / f"{image_path.stem}_R1_roi.png")
            anomaly_path = anomaly_dir / f"{image_path.stem}_anomaly.png"
            save_anomaly_map(roi_result["anomaly_map"], anomaly_path)

        if args.anomaly_threshold is None:
            decision = "UNCALIBRATED"
            final_result = predicted
            mark_label = predicted
        elif anomaly_score < args.anomaly_threshold:
            decision = "PASS"
            final_result = "PASS"
            mark_label = "PASS"
        else:
            decision = "NG"
            final_result = f"NG: {predicted}"
            mark_label = f"NG {predicted}"

        # marked/ is intentionally the clean final deliverable:
        # one primary bbox + one final result only. All intermediate candidates
        # remain available in all_regions, rois/, and full_heatmaps/ for debugging.
        marked_path = marked_dir / f"{image_path.stem}_marked.jpg"
        pipeline.save_full_image_overlay(
            image_path=image_path,
            bbox=primary_bbox,
            output_path=marked_path,
            label=mark_label,
            anomaly_score=anomaly_score,
            similarity=top1_similarity,
        )

        row = {
            "image": image_path.name,
            "image_path": str(image_path.resolve()),
            "inspection_mode": inspection_mode,
            "patchcore_anomaly_score": float(anomaly_score),
            "anomaly_decision": decision,
            "bbox_source": bbox_source,
            "num_regions": int(num_regions),
            "primary_bbox_x1": int(primary_bbox[0]),
            "primary_bbox_y1": int(primary_bbox[1]),
            "primary_bbox_x2": int(primary_bbox[2]),
            "primary_bbox_y2": int(primary_bbox[3]),
            "predicted_known_defect": predicted,
            "top1_similarity": float(top1_similarity),
            "margin": float(margin),
            "all_regions": all_regions_json,
            "final_result": final_result,
            "marked_image": str(marked_path.resolve()),
            "anomaly_map": str(Path(anomaly_path).resolve()),
        }
        rows.append(row)

        print(
            f"[{index:02d}/{len(images):02d}] {image_path.name} | "
            f"PatchCore={anomaly_score:.4f} | regions={num_regions} | "
            f"FINAL bbox={primary_bbox} | FINAL class={final_result} | "
            f"sim={top1_similarity:.4f} | margin={margin:.4f}"
        )

    if not rows:
        raise RuntimeError("No test image produced a usable anomaly region.")

    csv_path = output_dir / "results.csv"
    json_path = output_dir / "results.json"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nBatch inspection finished.")
    print(f"CSV:             {csv_path.resolve()}")
    print(f"JSON:            {json_path.resolve()}")
    print(f"Final marked:    {marked_dir.resolve()}")
    print(f"Region ROIs:     {roi_dir.resolve()}  (debug)")
    if inspection_mode == "tiled":
        print(f"Full heatmaps:   {full_heatmap_dir.resolve()}  (debug)")


if __name__ == "__main__":
    main()
