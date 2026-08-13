from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import FusedOpenSetRecognizer, calibrate_from_support
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-defect-class-out evaluation for Known/Unknown recognition. "
            "Thresholds are calibrated from each fold's support exemplars only."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/unknown_lodo",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument("--center-fraction", type=float, default=0.50)
    parser.add_argument("--support-quantile", type=float, default=0.10)
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

    if args.shots < 2:
        raise ValueError("--shots must be >= 2 for support-only leave-one-out calibration")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [p for p in test_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if len(class_dirs) < 3:
        raise RuntimeError("Need at least 3 defect classes for leave-one-class-out evaluation")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    records = []
    print("\n========== Precompute PatchCore ROIs + DINOv2 features ==========")
    for class_dir in class_dirs:
        files = image_files(class_dir)
        if len(files) <= args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(files)} images; need more than shots={args.shots}"
            )

        for index, image_path in enumerate(files):
            roi_result = pipeline.extract_roi(image_path)
            cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
            center_embedding = pipeline.embed_roi(
                roi_result["roi"],
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )

            roi_path = output_dir / "common_rois" / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            records.append(
                {
                    "class_name": class_dir.name,
                    "index_in_class": index,
                    "image_path": image_path,
                    "roi_path": roi_path.resolve(),
                    "anomaly_score": roi_result["anomaly_score"],
                    "bbox": roi_result["bbox"],
                    "cls_embedding": cls_embedding,
                    "center_embedding": center_embedding,
                }
            )
            print(
                f"{class_dir.name}/{image_path.name} "
                f"score={roi_result['anomaly_score']:.3f} bbox={roi_result['bbox']}"
            )

    class_names = [d.name for d in class_dirs]
    fold_summaries = []

    for held_out in class_names:
        fold_dir = output_dir / f"held_out_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        support_records = [
            r
            for r in records
            if r["class_name"] != held_out and r["index_in_class"] < args.shots
        ]
        known_query_records = [
            r
            for r in records
            if r["class_name"] != held_out and r["index_in_class"] >= args.shots
        ]
        unknown_query_records = [r for r in records if r["class_name"] == held_out]

        support_cls = np.stack([r["cls_embedding"] for r in support_records], axis=0)
        support_center = np.stack([r["center_embedding"] for r in support_records], axis=0)
        support_labels = [r["class_name"] for r in support_records]
        support_paths = [str(r["image_path"].resolve()) for r in support_records]

        cls_bank = DefectExemplarBank(support_cls, support_labels, support_paths)
        center_bank = DefectExemplarBank(support_center, support_labels, support_paths)
        calibration, diagnostics = calibrate_from_support(
            support_cls,
            support_center,
            support_labels,
            support_quantile=args.support_quantile,
            cls_weight=0.50,
            center_weight=0.50,
            center_fraction=args.center_fraction,
        )
        recognizer = FusedOpenSetRecognizer(cls_bank, center_bank, calibration)

        cls_bank.save(fold_dir / "bank_cls")
        center_bank.save(fold_dir / "bank_patch_center")
        calibration.save(fold_dir / "open_set_calibration.json")

        with open(
            fold_dir / "support_calibration_diagnostics.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(diagnostics[0].keys()))
            writer.writeheader()
            writer.writerows(diagnostics)

        rows = []
        known_total = 0
        known_accepted = 0
        known_exact_correct = 0
        known_accepted_correct = 0
        unknown_total = 0
        unknown_rejected = 0

        queries = [("known", r) for r in known_query_records] + [
            ("unknown", r) for r in unknown_query_records
        ]

        print(f"\n========== Held-out Unknown: {held_out} ==========")
        print(f"known classes: {cls_bank.classes}")
        print(f"support exemplars: {len(support_records)}")
        print(f"known queries: {len(known_query_records)}")
        print(f"unknown queries: {len(unknown_query_records)}")
        print(f"similarity threshold: {calibration.similarity_threshold:.6f}")
        print(f"margin threshold: {calibration.margin_threshold:.6f}")

        for query_index, (query_kind, record) in enumerate(queries, start=1):
            result = recognizer.predict_embeddings(
                record["cls_embedding"], record["center_embedding"]
            )
            true_class = record["class_name"]

            if query_kind == "unknown":
                unknown_total += 1
                correct = result["predicted_class"] == "Unknown"
                unknown_rejected += int(correct)
                expected = "Unknown"
            else:
                known_total += 1
                accepted = result["accepted_as_known"]
                known_accepted += int(accepted)
                accepted_correct = accepted and result["predicted_class"] == true_class
                known_accepted_correct += int(accepted_correct)
                correct = result["predicted_class"] == true_class
                known_exact_correct += int(correct)
                expected = true_class

            rows.append(
                {
                    "held_out_class": held_out,
                    "query_kind": query_kind,
                    "true_class": true_class,
                    "expected_output": expected,
                    "predicted_class": result["predicted_class"],
                    "nearest_known_class": result["nearest_known_class"],
                    "correct": int(correct),
                    "accepted_as_known": int(result["accepted_as_known"]),
                    "top1_similarity": f"{result['top1_similarity']:.6f}",
                    "similarity_threshold": f"{result['similarity_threshold']:.6f}",
                    "margin": f"{result['margin']:.6f}",
                    "margin_threshold": f"{result['margin_threshold']:.6f}",
                    "top2_class": result["top2_class"] or "",
                    "image_path": str(record["image_path"]),
                    "roi_path": str(record["roi_path"]),
                }
            )

            print(
                f"[{query_index:03d}/{len(queries):03d}] {query_kind:<7} "
                f"{true_class}/{record['image_path'].name} => {result['predicted_class']} "
                f"near={result['nearest_known_class']} sim={result['top1_similarity']:.4f} "
                f"margin={result['margin']:.4f} {'OK' if correct else 'MISS'}"
            )

        results_csv = fold_dir / "open_set_results.csv"
        with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        total = known_total + unknown_total
        open_set_correct = known_exact_correct + unknown_rejected
        known_accept_rate = known_accepted / max(1, known_total)
        known_exact_accuracy = known_exact_correct / max(1, known_total)
        accepted_known_accuracy = known_accepted_correct / max(1, known_accepted)
        unknown_rejection_rate = unknown_rejected / max(1, unknown_total)
        open_set_accuracy = open_set_correct / max(1, total)

        fold_summary = {
            "held_out_class": held_out,
            "similarity_threshold": calibration.similarity_threshold,
            "margin_threshold": calibration.margin_threshold,
            "known_total": known_total,
            "known_accept_rate": known_accept_rate,
            "known_exact_accuracy": known_exact_accuracy,
            "accepted_known_accuracy": accepted_known_accuracy,
            "unknown_total": unknown_total,
            "unknown_rejection_rate": unknown_rejection_rate,
            "open_set_accuracy": open_set_accuracy,
        }
        fold_summaries.append(fold_summary)

        print(f"known accept rate: {known_accept_rate:.2%}")
        print(f"known exact accuracy: {known_exact_accuracy:.2%}")
        print(f"accepted-known accuracy: {accepted_known_accuracy:.2%}")
        print(f"Unknown rejection rate: {unknown_rejection_rate:.2%}")
        print(f"open-set accuracy: {open_set_accuracy:.2%}")

    summary_csv = output_dir / "lodo_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(fold_summaries[0].keys()))
        writer.writeheader()
        writer.writerows(fold_summaries)

    mean_known_accept = float(np.mean([r["known_accept_rate"] for r in fold_summaries]))
    mean_known_exact = float(np.mean([r["known_exact_accuracy"] for r in fold_summaries]))
    mean_unknown_reject = float(
        np.mean([r["unknown_rejection_rate"] for r in fold_summaries])
    )
    mean_open_set = float(np.mean([r["open_set_accuracy"] for r in fold_summaries]))

    print("\n=============== Unknown LODO 汇总 ===============")
    for row in fold_summaries:
        print(
            f"held_out={row['held_out_class']:<20} "
            f"known_accept={row['known_accept_rate']:>7.2%} "
            f"known_exact={row['known_exact_accuracy']:>7.2%} "
            f"unknown_reject={row['unknown_rejection_rate']:>7.2%} "
            f"open_set={row['open_set_accuracy']:>7.2%}"
        )
    print("--------------------------------------------------")
    print(f"mean known accept rate: {mean_known_accept:.2%}")
    print(f"mean known exact accuracy: {mean_known_exact:.2%}")
    print(f"mean Unknown rejection rate: {mean_unknown_reject:.2%}")
    print(f"mean open-set accuracy: {mean_open_set:.2%}")
    print(f"summary CSV: {summary_csv.resolve()}")
    print("==================================================")


if __name__ == "__main__":
    main()
