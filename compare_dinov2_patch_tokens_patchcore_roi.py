from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import FEATURE_MODES
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_MODES = ("cls", "patch_mean", "patch_center")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Single-factor comparison of DINOv2 CLS, all-patch mean, and center-patch "
            "mean embeddings using exactly the same PatchCore ROIs and 3-shot split."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/dinov2_patch_token_compare",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument(
        "--center-fraction",
        type=float,
        default=0.50,
        help="Fraction of the square patch-token grid retained per side for patch_center.",
    )
    parser.add_argument(
        "--feature-modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=list(FEATURE_MODES),
    )
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def evaluate_mode(
    mode: str,
    pipeline: PatchCoreDINOv2Pipeline,
    roi_records: list[dict],
    output_dir: Path,
    center_fraction: float,
):
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)

    support_records = [r for r in roi_records if r["split"] == "support"]
    query_records = [r for r in roi_records if r["split"] == "query"]

    support_embeddings = []
    support_labels = []
    support_paths = []

    print(f"\n========== Build bank: {mode} ==========")
    for index, record in enumerate(support_records, start=1):
        embedding = pipeline.embed_roi(
            record["roi"],
            feature_mode=mode,
            center_fraction=center_fraction,
        )
        support_embeddings.append(embedding)
        support_labels.append(record["true_class"])
        support_paths.append(str(record["image_path"].resolve()))
        print(
            f"[{index:02d}/{len(support_records):02d}] "
            f"{record['true_class']}/{record['image_path'].name}"
        )

    bank = DefectExemplarBank(
        np.stack(support_embeddings, axis=0),
        support_labels,
        support_paths,
    )
    bank.save(mode_dir / "bank")

    rows = []
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)

    print(f"\n========== Evaluate: {mode} ==========")
    for index, record in enumerate(query_records, start=1):
        embedding = pipeline.embed_roi(
            record["roi"],
            feature_mode=mode,
            center_fraction=center_fraction,
        )
        result = bank.predict_embedding(embedding)

        true_class = record["true_class"]
        pred = result["predicted_class"]
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
                "top1_similarity": f"{result['top1_similarity']:.6f}",
                "top2_class": result["top2_class"] or "",
                "top2_similarity": f"{result['top2_similarity']:.6f}",
                "margin": f"{result['margin']:.6f}",
                "nearest_exemplar": result["nearest_exemplar"],
                "feature_mode": mode,
                "center_fraction": f"{center_fraction:.4f}",
            }
        )

        print(
            f"[{index:03d}/{len(query_records):03d}] "
            f"{true_class}/{record['image_path'].name} => {pred} "
            f"sim={result['top1_similarity']:.4f} "
            f"margin={result['margin']:.4f} "
            f"{'OK' if correct else 'MISS'}"
        )

    results_csv = mode_dir / "fewshot_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    classes = sorted(set(bank.classes) | set(per_class_total.keys()))
    confusion_csv = mode_dir / "confusion_matrix.csv"
    with open(confusion_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for true_class in classes:
            writer.writerow([true_class, *[confusion[true_class][p] for p in classes]])

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())
    overall = correct / max(1, total)

    class_accuracy = {
        name: per_class_correct[name] / max(1, per_class_total[name]) for name in classes
    }

    summary_csv = mode_dir / "class_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for name in classes:
            writer.writerow(
                [
                    name,
                    per_class_correct[name],
                    per_class_total[name],
                    f"{class_accuracy[name]:.6f}",
                ]
            )

    print(f"\n=============== {mode} 汇总 ===============")
    print(f"Top-1 accuracy: {overall:.2%} ({correct}/{total})")
    for name in classes:
        print(
            f"{name:<20} {per_class_correct[name]:>3}/{per_class_total[name]:<3} "
            f"= {class_accuracy[name]:.2%}"
        )
    print("===========================================")

    return {
        "feature_mode": mode,
        "overall_accuracy": overall,
        "correct": correct,
        "total": total,
        **{f"{name}_accuracy": class_accuracy[name] for name in classes},
    }


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

    roi_records = []
    print("\n========== Build common PatchCore ROIs once ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images; need more than shots={args.shots}"
            )

        for index, image_path in enumerate(images):
            split = "support" if index < args.shots else "query"
            roi_result = pipeline.extract_roi(image_path)

            roi_path = output_dir / "common_rois" / split / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            overlay_path = output_dir / "common_overlays" / split / class_dir.name / image_path.name
            pipeline.save_bbox_overlay(
                roi_result["display_image"], roi_result["bbox"], overlay_path
            )

            roi_records.append(
                {
                    "split": split,
                    "true_class": class_dir.name,
                    "image_path": image_path,
                    "roi_path": roi_path.resolve(),
                    "roi": roi_result["roi"],
                    "bbox": roi_result["bbox"],
                    "anomaly_score": roi_result["anomaly_score"],
                    "bbox_source": roi_result["bbox_source"],
                }
            )

            print(
                f"{split.upper():7} {class_dir.name}/{image_path.name} "
                f"score={roi_result['anomaly_score']:.3f} "
                f"bbox={roi_result['bbox']}"
            )

    summaries = []
    for mode in args.feature_modes:
        summaries.append(
            evaluate_mode(
                mode=mode,
                pipeline=pipeline,
                roi_records=roi_records,
                output_dir=output_dir,
                center_fraction=args.center_fraction,
            )
        )

    summary_csv = output_dir / "feature_comparison_summary.csv"
    all_fields = ["feature_mode", "overall_accuracy", "correct", "total"]
    class_fields = sorted(
        {key for row in summaries for key in row.keys() if key.endswith("_accuracy") and key != "overall_accuracy"}
    )
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields + class_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=============== DINOv2 特征模式汇总 ===============")
    for row in summaries:
        thread_side = row.get("thread_side_accuracy", 0.0)
        scratch_head = row.get("scratch_head_accuracy", 0.0)
        thread_top = row.get("thread_top_accuracy", 0.0)
        print(
            f"{row['feature_mode']:<14} | overall={row['overall_accuracy']:>7.2%} "
            f"| thread_side={thread_side:>7.2%} "
            f"| scratch_head={scratch_head:>7.2%} "
            f"| thread_top={thread_top:>7.2%}"
        )
    print(f"summary CSV: {summary_csv.resolve()}")
    print("====================================================")


if __name__ == "__main__":
    main()
