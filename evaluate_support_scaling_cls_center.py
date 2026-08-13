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
    p = argparse.ArgumentParser(
        description=(
            "Evaluate whether more few-shot defect exemplars improve CLS + Patch Center "
            "classification while keeping the exact same query set across shot counts."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/screw/support_scaling_cls_center",
    )
    p.add_argument("--shot-counts", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument(
        "--query-start",
        type=int,
        default=10,
        help=(
            "0-based index at which the shared query set starts. Default 10 means "
            "the first 10 images/class are reserved for nested support pools and all "
            "methods are evaluated only on images 11+."
        ),
    )
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
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shot_counts = sorted(set(int(v) for v in args.shot_counts))
    if not shot_counts or min(shot_counts) <= 0:
        raise ValueError("--shot-counts must contain positive integers")
    if args.query_start <= 0:
        raise ValueError("--query-start must be > 0")
    if max(shot_counts) > args.query_start:
        raise ValueError(
            "For a fixed non-overlapping query set, max(shot_counts) must be <= query_start"
        )
    if not (0.0 < args.center_fraction <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    class_dirs = sorted(
        [d for d in test_dir.iterdir() if d.is_dir() and d.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if not class_dirs:
        raise RuntimeError(f"No defect classes found in {test_dir}")

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    records: list[dict] = []
    print("\n========== Precompute common PatchCore ROI + DINOv2 CLS/Center ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.query_start:
            raise RuntimeError(
                f"{class_dir.name}: {len(images)} images, need > query_start={args.query_start}"
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
                    "cls_embedding": cls_embedding,
                    "center_embedding": center_embedding,
                    "anomaly_score": roi_result["anomaly_score"],
                }
            )
            role = "QUERY" if index >= args.query_start else "POOL"
            print(
                f"{role:5} {class_dir.name}/{image_path.name} "
                f"idx={index:02d} score={roi_result['anomaly_score']:.3f}"
            )

    classes = [d.name for d in class_dirs]
    shared_query = [r for r in records if r["class_index"] >= args.query_start]
    print(
        f"\nShared query set: {len(shared_query)} images, "
        f"starting at class index {args.query_start} (0-based)."
    )

    overall_rows: list[dict] = []

    for shots in shot_counts:
        support = [r for r in records if r["class_index"] < shots]
        labels = [r["true_class"] for r in support]
        paths = [str(r["image_path"].resolve()) for r in support]

        cls_bank = DefectExemplarBank(
            np.stack([r["cls_embedding"] for r in support], axis=0), labels, paths
        )
        center_bank = DefectExemplarBank(
            np.stack([r["center_embedding"] for r in support], axis=0), labels, paths
        )
        if cls_bank.classes != center_bank.classes:
            raise RuntimeError("CLS and center banks have different classes")

        totals = Counter()
        correct_cls = Counter()
        correct_center = Counter()
        correct_fusion = Counter()
        confusion_fusion = defaultdict(Counter)
        result_rows: list[dict] = []

        print(f"\n========== Evaluate shots={shots} on fixed query ==========")
        for i, r in enumerate(shared_query, 1):
            cls_result = cls_bank.predict_embedding(r["cls_embedding"])
            center_result = center_bank.predict_embedding(r["center_embedding"])
            fused_scores = {
                c: 0.5 * cls_result["class_scores"][c]
                + 0.5 * center_result["class_scores"][c]
                for c in cls_bank.classes
            }
            fused = rank_scores(fused_scores)

            true = r["true_class"]
            cls_pred = cls_result["predicted_class"]
            center_pred = center_result["predicted_class"]
            fusion_pred = fused["predicted_class"]

            totals[true] += 1
            correct_cls[true] += int(cls_pred == true)
            correct_center[true] += int(center_pred == true)
            correct_fusion[true] += int(fusion_pred == true)
            confusion_fusion[true][fusion_pred] += 1

            result_rows.append(
                {
                    "shots": shots,
                    "true_class": true,
                    "image_path": str(r["image_path"]),
                    "cls_pred": cls_pred,
                    "cls_top1": f"{cls_result['top1_similarity']:.6f}",
                    "center_pred": center_pred,
                    "center_top1": f"{center_result['top1_similarity']:.6f}",
                    "fusion_pred": fusion_pred,
                    "fusion_top1": f"{fused['top1_similarity']:.6f}",
                    "fusion_margin": f"{fused['margin']:.6f}",
                    "fusion_correct": int(fusion_pred == true),
                }
            )

            print(
                f"[{i:03d}/{len(shared_query):03d}] {true}/{r['image_path'].name} "
                f"CLS={cls_pred} Center={center_pred} Fusion={fusion_pred} "
                f"{'OK' if fusion_pred == true else 'MISS'}"
            )

        shot_dir = output_dir / f"shots_{shots}"
        write_rows(shot_dir / "results.csv", result_rows)

        class_summary = []
        for c in cls_bank.classes:
            n = totals[c]
            class_summary.append(
                {
                    "class": c,
                    "total": n,
                    "cls_correct": correct_cls[c],
                    "cls_accuracy": correct_cls[c] / max(1, n),
                    "center_correct": correct_center[c],
                    "center_accuracy": correct_center[c] / max(1, n),
                    "fusion_correct": correct_fusion[c],
                    "fusion_accuracy": correct_fusion[c] / max(1, n),
                }
            )
        write_rows(shot_dir / "class_summary.csv", class_summary)

        with open(
            shot_dir / "confusion_matrix_fusion.csv",
            "w",
            newline="",
            encoding="utf-8-sig",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *cls_bank.classes])
            for true_class in cls_bank.classes:
                writer.writerow(
                    [true_class, *[confusion_fusion[true_class][p] for p in cls_bank.classes]]
                )

        total = sum(totals.values())
        cls_correct_total = sum(correct_cls.values())
        center_correct_total = sum(correct_center.values())
        fusion_correct_total = sum(correct_fusion.values())

        row = {
            "shots": shots,
            "support_total": len(support),
            "shared_query_total": total,
            "cls_accuracy": cls_correct_total / max(1, total),
            "center_accuracy": center_correct_total / max(1, total),
            "fusion_accuracy": fusion_correct_total / max(1, total),
        }
        for c in cls_bank.classes:
            row[f"fusion_{c}_accuracy"] = correct_fusion[c] / max(1, totals[c])
        overall_rows.append(row)

    summary_csv = output_dir / "support_scaling_summary.csv"
    write_rows(summary_csv, overall_rows)

    print("\n=============== Fixed-Query Support Scaling 汇总 ===============")
    print(
        f"shared query start index: {args.query_start} (0-based), "
        f"shared query total={len(shared_query)}"
    )
    for row in overall_rows:
        thread_side = row.get("fusion_thread_side_accuracy", float("nan"))
        print(
            f"shots={row['shots']:>2} | "
            f"CLS={row['cls_accuracy']:.2%} | "
            f"Center={row['center_accuracy']:.2%} | "
            f"Fusion={row['fusion_accuracy']:.2%} | "
            f"thread_side={thread_side:.2%}"
        )
    print(f"summary CSV: {summary_csv.resolve()}")
    print("===============================================================")


if __name__ == "__main__":
    main()
