from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate equal-weight score fusion of DINOv2 CLS and center patch-token "
            "features on the same PatchCore ROIs and 3-shot split."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/dinov2_cls_center_fusion",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument("--center-fraction", type=float, default=0.50)
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if not (0.0 < args.center_fraction <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [p for p in test_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if not class_dirs:
        raise RuntimeError(f"No defect class directories found in {test_dir}")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    support_records = []
    query_records = []

    print("\n========== Build common PatchCore ROIs ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images; need more than shots={args.shots}"
            )

        for index, image_path in enumerate(images):
            split = "support" if index < args.shots else "query"
            roi_result = pipeline.extract_roi(image_path)

            roi_path = output_dir / "rois" / split / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            record = {
                "true_class": class_dir.name,
                "image_path": image_path,
                "roi_path": roi_path.resolve(),
                "roi": roi_result["roi"],
                "anomaly_score": roi_result["anomaly_score"],
                "bbox": roi_result["bbox"],
            }
            if split == "support":
                support_records.append(record)
            else:
                query_records.append(record)

            print(
                f"{split.upper():7} {class_dir.name}/{image_path.name} "
                f"score={roi_result['anomaly_score']:.3f} bbox={roi_result['bbox']}"
            )

    cls_embeddings = []
    center_embeddings = []
    labels = []
    support_paths = []

    print("\n========== Build CLS + Center banks ==========")
    for index, record in enumerate(support_records, start=1):
        cls_embeddings.append(
            pipeline.embed_roi(record["roi"], feature_mode="cls")
        )
        center_embeddings.append(
            pipeline.embed_roi(
                record["roi"],
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )
        )
        labels.append(record["true_class"])
        support_paths.append(str(record["image_path"].resolve()))
        print(
            f"[{index:02d}/{len(support_records):02d}] "
            f"{record['true_class']}/{record['image_path'].name}"
        )

    cls_bank = DefectExemplarBank(
        np.stack(cls_embeddings, axis=0), labels, support_paths
    )
    center_bank = DefectExemplarBank(
        np.stack(center_embeddings, axis=0), labels, support_paths
    )
    cls_bank.save(output_dir / "bank_cls")
    center_bank.save(output_dir / "bank_patch_center")

    if cls_bank.classes != center_bank.classes:
        raise RuntimeError("CLS and center banks have different class sets")
    classes = cls_bank.classes

    rows = []
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)

    print("\n========== Evaluate equal-weight score fusion ==========")
    for index, record in enumerate(query_records, start=1):
        cls_embedding = pipeline.embed_roi(record["roi"], feature_mode="cls")
        center_embedding = pipeline.embed_roi(
            record["roi"],
            feature_mode="patch_center",
            center_fraction=args.center_fraction,
        )

        cls_result = cls_bank.predict_embedding(cls_embedding)
        center_result = center_bank.predict_embedding(center_embedding)

        fused_scores = {
            class_name: 0.5 * cls_result["class_scores"][class_name]
            + 0.5 * center_result["class_scores"][class_name]
            for class_name in classes
        }
        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        pred, top1 = ranked[0]
        top2_class, top2 = ranked[1] if len(ranked) > 1 else ("", float("-inf"))
        margin = float(top1 - top2) if np.isfinite(top2) else float("inf")

        true_class = record["true_class"]
        correct = pred == true_class
        per_class_total[true_class] += 1
        per_class_correct[true_class] += int(correct)
        confusion[true_class][pred] += 1

        rows.append(
            {
                "true_class": true_class,
                "predicted_class": pred,
                "correct": int(correct),
                "image_path": str(record["image_path"]),
                "roi_path": str(record["roi_path"]),
                "anomaly_score": f"{record['anomaly_score']:.6f}",
                "bbox": ",".join(str(v) for v in record["bbox"]),
                "fused_top1_similarity": f"{top1:.6f}",
                "fused_top2_class": top2_class,
                "fused_top2_similarity": f"{top2:.6f}",
                "fused_margin": f"{margin:.6f}",
                "cls_pred": cls_result["predicted_class"],
                "cls_top1": f"{cls_result['top1_similarity']:.6f}",
                "center_pred": center_result["predicted_class"],
                "center_top1": f"{center_result['top1_similarity']:.6f}",
            }
        )

        print(
            f"[{index:03d}/{len(query_records):03d}] "
            f"{true_class}/{record['image_path'].name} => {pred} "
            f"fused={top1:.4f} margin={margin:.4f} "
            f"cls={cls_result['predicted_class']} center={center_result['predicted_class']} "
            f"{'OK' if correct else 'MISS'}"
        )

    results_csv = output_dir / "fewshot_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    confusion_csv = output_dir / "confusion_matrix.csv"
    with open(confusion_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for true_class in classes:
            writer.writerow([true_class, *[confusion[true_class][p] for p in classes]])

    summary_csv = output_dir / "class_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for class_name in classes:
            n = per_class_total[class_name]
            c = per_class_correct[class_name]
            writer.writerow([class_name, c, n, f"{c / max(1, n):.6f}"])

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())

    print("\n=============== CLS + Patch Center 50/50 融合汇总 ===============")
    print(f"Top-1 accuracy: {correct / max(1, total):.2%} ({correct}/{total})")
    for class_name in classes:
        n = per_class_total[class_name]
        c = per_class_correct[class_name]
        print(f"{class_name:<20} {c:>3}/{n:<3} = {c / max(1, n):.2%}")
    print(f"results: {results_csv.resolve()}")
    print(f"class summary: {summary_csv.resolve()}")
    print(f"confusion matrix: {confusion_csv.resolve()}")
    print("================================================================")


if __name__ == "__main__":
    main()
