from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import FusedOpenSetRecognizer
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline
from app.defect.pseudo_unknown_calibration import calibrate_with_pseudo_unknown


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Leave-one-defect-class-out Known/Unknown evaluation using support-only "
            "class-held-out pseudo-Unknown calibration."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/unknown_lodo_pseudo_calibrated",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    parser.add_argument("--center-fraction", type=float, default=0.50)
    return parser.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def write_dict_rows(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.shots < 2:
        raise ValueError("--shots must be >= 2 for support-only calibration")
    if not (0.0 < args.center_fraction <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [p for p in test_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if len(class_dirs) < 3:
        raise RuntimeError("LODO evaluation needs at least 3 defect classes")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    # Extract each PatchCore ROI and both DINOv2 features once. Every fold uses
    # exactly the same image representation; only the class bank changes.
    records: list[dict] = []
    print("\n========== Precompute common PatchCore ROI + DINOv2 features ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images; need more than shots={args.shots}"
            )

        for index, image_path in enumerate(images):
            roi_result = pipeline.extract_roi(image_path)
            cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
            center_embedding = pipeline.embed_roi(
                roi_result["roi"],
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )

            records.append(
                {
                    "true_class": class_dir.name,
                    "class_index": index,
                    "image_path": image_path,
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

    classes = [d.name for d in class_dirs]
    summaries: list[dict] = []

    for held_out in classes:
        fold_dir = output_dir / f"held_out_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        support = [
            r
            for r in records
            if r["true_class"] != held_out and r["class_index"] < args.shots
        ]
        known_query = [
            r
            for r in records
            if r["true_class"] != held_out and r["class_index"] >= args.shots
        ]
        unknown_query = [r for r in records if r["true_class"] == held_out]

        support_cls = np.stack([r["cls_embedding"] for r in support], axis=0)
        support_center = np.stack([r["center_embedding"] for r in support], axis=0)
        support_labels = [r["true_class"] for r in support]
        support_paths = [str(r["image_path"].resolve()) for r in support]

        cls_bank = DefectExemplarBank(support_cls, support_labels, support_paths)
        center_bank = DefectExemplarBank(support_center, support_labels, support_paths)

        calibration, calibration_records, calibration_summary = calibrate_with_pseudo_unknown(
            cls_embeddings=support_cls,
            center_embeddings=support_center,
            labels=support_labels,
            cls_weight=0.50,
            center_weight=0.50,
            center_fraction=args.center_fraction,
        )
        recognizer = FusedOpenSetRecognizer(cls_bank, center_bank, calibration)

        write_dict_rows(fold_dir / "support_calibration_records.csv", calibration_records)
        with open(fold_dir / "calibration_summary.json", "w", encoding="utf-8") as f:
            json.dump(calibration_summary.to_dict(), f, ensure_ascii=False, indent=2)

        rows = []
        known_accept = 0
        known_exact = 0
        unknown_reject = 0

        print(f"\n========== held_out = {held_out} ==========")
        print(
            "support calibration: "
            f"sim>={calibration.similarity_threshold:.6f}, "
            f"margin>={calibration.margin_threshold:.6f}, "
            f"pseudo_known_accept={calibration_summary.known_accept_rate:.2%}, "
            f"pseudo_unknown_reject={calibration_summary.pseudo_unknown_reject_rate:.2%}"
        )

        eval_records = [("known", r) for r in known_query] + [
            ("unknown", r) for r in unknown_query
        ]

        for scenario, record in eval_records:
            result = recognizer.predict_embeddings(
                record["cls_embedding"], record["center_embedding"]
            )
            if scenario == "known":
                accepted = bool(result["accepted_as_known"])
                exact = accepted and result["predicted_class"] == record["true_class"]
                known_accept += int(accepted)
                known_exact += int(exact)
                correct_open_set = exact
            else:
                rejected = not bool(result["accepted_as_known"])
                unknown_reject += int(rejected)
                correct_open_set = rejected

            rows.append(
                {
                    "held_out": held_out,
                    "scenario": scenario,
                    "true_class": record["true_class"],
                    "image_path": str(record["image_path"]),
                    "predicted_class": result["predicted_class"],
                    "nearest_known_class": result["nearest_known_class"],
                    "accepted_as_known": int(result["accepted_as_known"]),
                    "open_set_correct": int(correct_open_set),
                    "top1_similarity": f"{result['top1_similarity']:.6f}",
                    "margin": f"{result['margin']:.6f}",
                    "similarity_threshold": f"{result['similarity_threshold']:.6f}",
                    "margin_threshold": f"{result['margin_threshold']:.6f}",
                    "anomaly_score": f"{record['anomaly_score']:.6f}",
                    "bbox": ",".join(str(v) for v in record["bbox"]),
                }
            )

        write_dict_rows(fold_dir / "results.csv", rows)

        known_total = len(known_query)
        unknown_total = len(unknown_query)
        eval_total = known_total + unknown_total
        known_accept_rate = known_accept / max(1, known_total)
        known_exact_rate = known_exact / max(1, known_total)
        unknown_reject_rate = unknown_reject / max(1, unknown_total)
        open_set_accuracy = (known_exact + unknown_reject) / max(1, eval_total)

        summary = {
            "held_out": held_out,
            "known_total": known_total,
            "unknown_total": unknown_total,
            "known_accept_rate": known_accept_rate,
            "known_exact_accuracy": known_exact_rate,
            "unknown_reject_rate": unknown_reject_rate,
            "open_set_accuracy": open_set_accuracy,
            "similarity_threshold": calibration.similarity_threshold,
            "margin_threshold": calibration.margin_threshold,
            "support_pseudo_known_accept": calibration_summary.known_accept_rate,
            "support_pseudo_unknown_reject": calibration_summary.pseudo_unknown_reject_rate,
            "support_balanced_accuracy": calibration_summary.balanced_accuracy,
        }
        summaries.append(summary)

        print(
            f"held_out={held_out:<20} "
            f"known_accept={known_accept_rate:>7.2%} "
            f"known_exact={known_exact_rate:>7.2%} "
            f"unknown_reject={unknown_reject_rate:>7.2%} "
            f"open_set={open_set_accuracy:>7.2%}"
        )

    summary_csv = output_dir / "lodo_summary.csv"
    write_dict_rows(summary_csv, summaries)

    mean_known_accept = float(np.mean([r["known_accept_rate"] for r in summaries]))
    mean_known_exact = float(np.mean([r["known_exact_accuracy"] for r in summaries]))
    mean_unknown_reject = float(np.mean([r["unknown_reject_rate"] for r in summaries]))
    mean_open_set = float(np.mean([r["open_set_accuracy"] for r in summaries]))

    print("\n=============== Pseudo-Unknown Calibrated LODO 汇总 ===============")
    for row in summaries:
        print(
            f"held_out={row['held_out']:<20} "
            f"known_accept={row['known_accept_rate']:>7.2%} "
            f"known_exact={row['known_exact_accuracy']:>7.2%} "
            f"unknown_reject={row['unknown_reject_rate']:>7.2%} "
            f"open_set={row['open_set_accuracy']:>7.2%}"
        )
    print("-----------------------------------------------------------------")
    print(f"mean known accept rate: {mean_known_accept:.2%}")
    print(f"mean known exact accuracy: {mean_known_exact:.2%}")
    print(f"mean Unknown rejection rate: {mean_unknown_reject:.2%}")
    print(f"mean open-set accuracy: {mean_open_set:.2%}")
    print(f"summary CSV: {summary_csv.resolve()}")
    print("=================================================================")


if __name__ == "__main__":
    main()
