from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import FusedOpenSetRecognizer, calibrate_from_support
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Fixed-query leave-one-defect-class-out open-set evaluation across "
            "3/5/10-shot support counts. Thresholds are calibrated from support "
            "leave-one-out only; query labels are evaluation-only."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/screw/unknown_lodo_support_scaling",
    )
    p.add_argument("--shot-counts", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument(
        "--query-start",
        type=int,
        default=10,
        help=(
            "0-based known-query start shared across all shot counts. Default 10 "
            "means first 10 images/class are reserved as nested support pool and "
            "known queries are images 11+."
        ),
    )
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument("--support-quantile", type=float, default=0.10)
    return p.parse_args()


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def write_rows(path: Path, rows: list[dict]) -> None:
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
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shot_counts = sorted(set(args.shot_counts))
    if not shot_counts or min(shot_counts) < 2:
        raise ValueError("shot counts must all be >= 2")
    if max(shot_counts) > args.query_start:
        raise ValueError("max shot count cannot exceed query-start in fixed-query evaluation")
    if not (0.0 <= args.support_quantile < 0.5):
        raise ValueError("support-quantile must be in [0,0.5)")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [d for d in test_dir.iterdir() if d.is_dir() and d.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if len(class_dirs) < 3:
        raise RuntimeError("Need at least 3 defect classes for LODO evaluation")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    records: list[dict] = []
    print("\n========== Precompute PatchCore ROI + CLS + Patch Center ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.query_start:
            raise RuntimeError(
                f"{class_dir.name}: needs more than query-start={args.query_start} images"
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
                    "class_name": class_dir.name,
                    "index_in_class": index,
                    "image_path": image_path,
                    "cls_embedding": cls_embedding,
                    "center_embedding": center_embedding,
                }
            )
            print(f"{class_dir.name}/{image_path.name}")

    class_names = [d.name for d in class_dirs]
    fold_rows: list[dict] = []
    result_rows: list[dict] = []

    for shots in shot_counts:
        print(f"\n================ shots={shots} ================")
        for held_out in class_names:
            support = [
                r
                for r in records
                if r["class_name"] != held_out and r["index_in_class"] < shots
            ]
            known_queries = [
                r
                for r in records
                if r["class_name"] != held_out
                and r["index_in_class"] >= args.query_start
            ]
            unknown_queries = [r for r in records if r["class_name"] == held_out]

            support_cls = np.stack([r["cls_embedding"] for r in support], axis=0)
            support_center = np.stack([r["center_embedding"] for r in support], axis=0)
            support_labels = [r["class_name"] for r in support]
            support_paths = [str(r["image_path"].resolve()) for r in support]

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
            loo_accuracy = float(np.mean([d["loo_correct"] for d in diagnostics]))

            known_total = 0
            known_accepted = 0
            known_exact = 0
            known_accepted_correct = 0
            unknown_total = 0
            unknown_rejected = 0

            for query_kind, group in (("known", known_queries), ("unknown", unknown_queries)):
                for record in group:
                    result = recognizer.predict_embeddings(
                        record["cls_embedding"], record["center_embedding"]
                    )
                    true_class = record["class_name"]

                    if query_kind == "known":
                        known_total += 1
                        accepted = bool(result["accepted_as_known"])
                        known_accepted += int(accepted)
                        exact = result["predicted_class"] == true_class
                        known_exact += int(exact)
                        known_accepted_correct += int(accepted and exact)
                        correct = exact
                    else:
                        unknown_total += 1
                        correct = result["predicted_class"] == "Unknown"
                        unknown_rejected += int(correct)

                    result_rows.append(
                        {
                            "shots": shots,
                            "held_out_class": held_out,
                            "query_kind": query_kind,
                            "true_class": true_class,
                            "image_path": str(record["image_path"]),
                            "predicted_class": result["predicted_class"],
                            "nearest_known_class": result["nearest_known_class"],
                            "correct": int(correct),
                            "accepted_as_known": int(result["accepted_as_known"]),
                            "top1_similarity": result["top1_similarity"],
                            "similarity_threshold": result["similarity_threshold"],
                            "margin": result["margin"],
                            "margin_threshold": result["margin_threshold"],
                        }
                    )

            known_accept_rate = known_accepted / max(1, known_total)
            known_exact_accuracy = known_exact / max(1, known_total)
            accepted_known_accuracy = known_accepted_correct / max(1, known_accepted)
            unknown_rejection_rate = unknown_rejected / max(1, unknown_total)
            open_set_accuracy = (known_exact + unknown_rejected) / max(
                1, known_total + unknown_total
            )

            fold = {
                "shots": shots,
                "held_out_class": held_out,
                "support_count": len(support),
                "known_query_total": known_total,
                "unknown_query_total": unknown_total,
                "support_loo_accuracy": loo_accuracy,
                "similarity_threshold": calibration.similarity_threshold,
                "margin_threshold": calibration.margin_threshold,
                "known_accept_rate": known_accept_rate,
                "known_exact_accuracy": known_exact_accuracy,
                "accepted_known_accuracy": accepted_known_accuracy,
                "unknown_rejection_rate": unknown_rejection_rate,
                "open_set_accuracy": open_set_accuracy,
            }
            fold_rows.append(fold)

            print(
                f"held_out={held_out:<20} "
                f"LOO={loo_accuracy:>7.2%} "
                f"known_accept={known_accept_rate:>7.2%} "
                f"known_exact={known_exact_accuracy:>7.2%} "
                f"unknown_reject={unknown_rejection_rate:>7.2%} "
                f"open_set={open_set_accuracy:>7.2%}"
            )

    write_rows(out / "fold_summary.csv", fold_rows)
    write_rows(out / "results.csv", result_rows)

    aggregate_rows: list[dict] = []
    print("\n=============== Fixed-Query Unknown LODO Support Scaling 汇总 ===============")
    print(f"known query start index: {args.query_start} (0-based)")
    for shots in shot_counts:
        rows = [r for r in fold_rows if r["shots"] == shots]
        aggregate = {
            "shots": shots,
            "mean_support_loo_accuracy": float(np.mean([r["support_loo_accuracy"] for r in rows])),
            "mean_known_accept_rate": float(np.mean([r["known_accept_rate"] for r in rows])),
            "mean_known_exact_accuracy": float(np.mean([r["known_exact_accuracy"] for r in rows])),
            "mean_accepted_known_accuracy": float(np.mean([r["accepted_known_accuracy"] for r in rows])),
            "mean_unknown_rejection_rate": float(np.mean([r["unknown_rejection_rate"] for r in rows])),
            "mean_open_set_accuracy": float(np.mean([r["open_set_accuracy"] for r in rows])),
        }
        aggregate_rows.append(aggregate)
        print(
            f"shots={shots:>2} | "
            f"LOO={aggregate['mean_support_loo_accuracy']:.2%} | "
            f"known_accept={aggregate['mean_known_accept_rate']:.2%} | "
            f"known_exact={aggregate['mean_known_exact_accuracy']:.2%} | "
            f"unknown_reject={aggregate['mean_unknown_rejection_rate']:.2%} | "
            f"open_set={aggregate['mean_open_set_accuracy']:.2%}"
        )
    write_rows(out / "support_scaling_summary.csv", aggregate_rows)
    print(f"summary CSV: {(out/'support_scaling_summary.csv').resolve()}")
    print("============================================================================")


if __name__ == "__main__":
    main()
