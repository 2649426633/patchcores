from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.open_set_fusion import FusedOpenSetRecognizer, calibrate_from_support
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline
from app.defect.support_consistency_gate import (
    SupportConsistencyGate,
    calibrate_support_consistency,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate a support-only class-conditional Top-K consistency gate on top of "
            "the existing CLS+Center open-set recognizer. Known queries are fixed to "
            "images after query-start; held-out classes are Unknown."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/screw/unknown_consistency_gate",
    )
    p.add_argument("--shots", type=int, default=10)
    p.add_argument("--query-start", type=int, default=10)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--support-quantile", type=float, default=0.10)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
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
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def metrics_from_rows(rows: list[dict], pred_key: str, accepted_key: str) -> dict:
    known = [r for r in rows if r["query_kind"] == "known"]
    unknown = [r for r in rows if r["query_kind"] == "unknown"]

    known_accepted = sum(int(r[accepted_key]) for r in known)
    known_exact = sum(int(r[pred_key] == r["true_class"]) for r in known)
    known_accepted_correct = sum(
        int(r[accepted_key] and r[pred_key] == r["true_class"]) for r in known
    )
    unknown_rejected = sum(int(r[pred_key] == "Unknown") for r in unknown)
    total_correct = known_exact + unknown_rejected

    return {
        "known_total": len(known),
        "known_accept_rate": known_accepted / max(1, len(known)),
        "known_exact_accuracy": known_exact / max(1, len(known)),
        "accepted_known_accuracy": known_accepted_correct / max(1, known_accepted),
        "unknown_total": len(unknown),
        "unknown_rejection_rate": unknown_rejected / max(1, len(unknown)),
        "open_set_accuracy": total_correct / max(1, len(rows)),
    }


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.shots < 2:
        raise ValueError("--shots must be >= 2")
    if args.shots > args.query_start:
        raise ValueError("--shots cannot exceed --query-start for fixed-query evaluation")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [d for d in test_dir.iterdir() if d.is_dir() and d.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if len(class_dirs) < 3:
        raise RuntimeError("Need at least three defect classes")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    records: list[dict] = []
    print("\n========== Precompute PatchCore ROI + CLS + Center ==========")
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
    fold_summaries: list[dict] = []
    all_rows: list[dict] = []

    for held_out in class_names:
        fold_dir = out / f"held_out_{held_out}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        support = [
            r
            for r in records
            if r["class_name"] != held_out and r["index_in_class"] < args.shots
        ]
        known_query = [
            r
            for r in records
            if r["class_name"] != held_out and r["index_in_class"] >= args.query_start
        ]
        unknown_query = [r for r in records if r["class_name"] == held_out]

        support_cls = np.stack([r["cls_embedding"] for r in support], axis=0)
        support_center = np.stack([r["center_embedding"] for r in support], axis=0)
        support_labels = [r["class_name"] for r in support]
        support_paths = [str(r["image_path"].resolve()) for r in support]

        cls_bank = DefectExemplarBank(support_cls, support_labels, support_paths)
        center_bank = DefectExemplarBank(support_center, support_labels, support_paths)

        baseline_cal, baseline_diag = calibrate_from_support(
            support_cls,
            support_center,
            support_labels,
            support_quantile=args.support_quantile,
            cls_weight=0.50,
            center_weight=0.50,
            center_fraction=args.center_fraction,
        )
        baseline = FusedOpenSetRecognizer(cls_bank, center_bank, baseline_cal)

        consistency_cal, consistency_diag = calibrate_support_consistency(
            support_cls,
            support_center,
            support_labels,
            top_k=args.top_k,
            support_quantile=args.support_quantile,
            cls_weight=0.50,
            center_weight=0.50,
        )
        consistency_gate = SupportConsistencyGate(
            support_cls,
            support_center,
            support_labels,
            consistency_cal,
        )

        write_rows(fold_dir / "baseline_support_loo.csv", baseline_diag)
        write_rows(fold_dir / "consistency_support_loo.csv", consistency_diag)
        with open(fold_dir / "consistency_calibration.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "calibration_source": "support_leave_one_out_only",
                    "top_k": consistency_cal.top_k,
                    "support_quantile": consistency_cal.support_quantile,
                    "support_count": consistency_cal.support_count,
                    "class_thresholds": consistency_cal.class_thresholds,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        rows: list[dict] = []
        queries = [("known", r) for r in known_query] + [("unknown", r) for r in unknown_query]

        print(f"\n========== held_out={held_out} ==========")
        print(f"support={len(support)} known_query={len(known_query)} unknown={len(unknown_query)}")
        print(
            f"baseline thresholds: sim>={baseline_cal.similarity_threshold:.6f}, "
            f"margin>={baseline_cal.margin_threshold:.6f}"
        )
        print("consistency thresholds:")
        for class_name in sorted(consistency_cal.class_thresholds):
            print(
                f"  {class_name:<20} >= {consistency_cal.class_thresholds[class_name]:.6f}"
            )

        for query_kind, record in queries:
            base = baseline.predict_embeddings(
                record["cls_embedding"], record["center_embedding"]
            )
            density = consistency_gate.evaluate(
                record["cls_embedding"],
                record["center_embedding"],
                base["nearest_known_class"],
            )

            consistency_accepted = bool(
                base["accepted_as_known"] and density["consistency_ok"]
            )
            consistency_pred = (
                base["nearest_known_class"] if consistency_accepted else "Unknown"
            )

            row = {
                "held_out_class": held_out,
                "query_kind": query_kind,
                "true_class": record["class_name"],
                "image_path": str(record["image_path"]),
                "nearest_known_class": base["nearest_known_class"],
                "top1_similarity": float(base["top1_similarity"]),
                "margin": float(base["margin"]),
                "baseline_pred": base["predicted_class"],
                "baseline_accepted": int(base["accepted_as_known"]),
                "consistency_score": float(density["consistency_score"]),
                "consistency_threshold": float(density["consistency_threshold"]),
                "consistency_ok": int(density["consistency_ok"]),
                "consistency_pred": consistency_pred,
                "consistency_accepted": int(consistency_accepted),
            }
            rows.append(row)
            all_rows.append(row)

        baseline_metrics = metrics_from_rows(rows, "baseline_pred", "baseline_accepted")
        consistency_metrics = metrics_from_rows(
            rows, "consistency_pred", "consistency_accepted"
        )

        fold_summary = {
            "held_out_class": held_out,
            "shots": args.shots,
            "top_k": args.top_k,
            "baseline_known_accept": baseline_metrics["known_accept_rate"],
            "baseline_known_exact": baseline_metrics["known_exact_accuracy"],
            "baseline_unknown_reject": baseline_metrics["unknown_rejection_rate"],
            "baseline_open_set": baseline_metrics["open_set_accuracy"],
            "consistency_known_accept": consistency_metrics["known_accept_rate"],
            "consistency_known_exact": consistency_metrics["known_exact_accuracy"],
            "consistency_unknown_reject": consistency_metrics["unknown_rejection_rate"],
            "consistency_open_set": consistency_metrics["open_set_accuracy"],
        }
        fold_summaries.append(fold_summary)
        write_rows(fold_dir / "open_set_results.csv", rows)

        print(
            f"baseline:    known_accept={baseline_metrics['known_accept_rate']:.2%} "
            f"known_exact={baseline_metrics['known_exact_accuracy']:.2%} "
            f"unknown_reject={baseline_metrics['unknown_rejection_rate']:.2%} "
            f"open_set={baseline_metrics['open_set_accuracy']:.2%}"
        )
        print(
            f"consistency: known_accept={consistency_metrics['known_accept_rate']:.2%} "
            f"known_exact={consistency_metrics['known_exact_accuracy']:.2%} "
            f"unknown_reject={consistency_metrics['unknown_rejection_rate']:.2%} "
            f"open_set={consistency_metrics['open_set_accuracy']:.2%}"
        )

    write_rows(out / "all_results.csv", all_rows)
    write_rows(out / "lodo_summary.csv", fold_summaries)

    mean_keys = [
        "baseline_known_accept",
        "baseline_known_exact",
        "baseline_unknown_reject",
        "baseline_open_set",
        "consistency_known_accept",
        "consistency_known_exact",
        "consistency_unknown_reject",
        "consistency_open_set",
    ]
    means = {key: float(np.mean([r[key] for r in fold_summaries])) for key in mean_keys}

    print("\n=============== 10-shot Unknown Consistency Gate 汇总 ===============")
    for row in fold_summaries:
        print(
            f"held_out={row['held_out_class']:<20} "
            f"baseline(K={row['baseline_known_exact']:.2%}, U={row['baseline_unknown_reject']:.2%}, O={row['baseline_open_set']:.2%}) "
            f"consistency(K={row['consistency_known_exact']:.2%}, U={row['consistency_unknown_reject']:.2%}, O={row['consistency_open_set']:.2%})"
        )
    print("-----------------------------------------------------------------------")
    print(
        f"BASELINE    known_accept={means['baseline_known_accept']:.2%} "
        f"known_exact={means['baseline_known_exact']:.2%} "
        f"unknown_reject={means['baseline_unknown_reject']:.2%} "
        f"open_set={means['baseline_open_set']:.2%}"
    )
    print(
        f"CONSISTENCY known_accept={means['consistency_known_accept']:.2%} "
        f"known_exact={means['consistency_known_exact']:.2%} "
        f"unknown_reject={means['consistency_unknown_reject']:.2%} "
        f"open_set={means['consistency_open_set']:.2%}"
    )
    print(f"summary CSV: {(out/'lodo_summary.csv').resolve()}")
    print("=======================================================================")


if __name__ == "__main__":
    main()
