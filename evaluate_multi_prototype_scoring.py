from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from app.defect.hybrid_defect_bank import HybridDefectBank
from app.defect.multi_prototype_bank import MultiPrototypeDefectBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
MODES = ("exemplar", "mean_prototype", "multi_prototype_k2")


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare exemplar, single mean prototype and deterministic K=2 multi-prototype "
            "class scoring for CLS + Patch Center on an identical fixed query set."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/screw/multi_prototype_scoring",
    )
    p.add_argument("--shot-counts", type=int, nargs="+", default=[3, 5, 10])
    p.add_argument("--query-start", type=int, default=10)
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
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    shot_counts = sorted(set(args.shot_counts))
    if not shot_counts or min(shot_counts) <= 0:
        raise ValueError("shot counts must be positive")
    if max(shot_counts) > args.query_start:
        raise ValueError("max shot count cannot exceed query-start")

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

    records = []
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
                    "true_class": class_dir.name,
                    "class_index": index,
                    "image_path": image_path,
                    "cls_embedding": cls_embedding,
                    "center_embedding": center_embedding,
                }
            )
            print(f"{class_dir.name}/{image_path.name}")

    query = [r for r in records if r["class_index"] >= args.query_start]
    classes = sorted({r["true_class"] for r in records})
    summary_rows = []
    result_rows = []
    cluster_rows = []

    for shots in shot_counts:
        support = [r for r in records if r["class_index"] < shots]
        labels = [r["true_class"] for r in support]
        paths = [str(r["image_path"].resolve()) for r in support]
        cls_support = np.stack([r["cls_embedding"] for r in support])
        center_support = np.stack([r["center_embedding"] for r in support])

        cls_single = HybridDefectBank(cls_support, labels, paths)
        center_single = HybridDefectBank(center_support, labels, paths)
        cls_multi = MultiPrototypeDefectBank(cls_support, labels, prototypes_per_class=2)
        center_multi = MultiPrototypeDefectBank(center_support, labels, prototypes_per_class=2)

        for c in classes:
            cluster_rows.append(
                {
                    "shots": shots,
                    "feature": "cls",
                    "class": c,
                    "cluster_sizes": ",".join(map(str, cls_multi.prototype_info[c].cluster_sizes)),
                }
            )
            cluster_rows.append(
                {
                    "shots": shots,
                    "feature": "center",
                    "class": c,
                    "cluster_sizes": ",".join(map(str, center_multi.prototype_info[c].cluster_sizes)),
                }
            )

        correct = {mode: Counter() for mode in MODES}
        totals = Counter()
        print(f"\n========== shots={shots} ==========")

        for record in query:
            true = record["true_class"]
            totals[true] += 1

            for mode in MODES:
                if mode == "exemplar":
                    cls_result = cls_single.predict_embedding(record["cls_embedding"], mode="exemplar")
                    center_result = center_single.predict_embedding(record["center_embedding"], mode="exemplar")
                elif mode == "mean_prototype":
                    cls_result = cls_single.predict_embedding(record["cls_embedding"], mode="prototype")
                    center_result = center_single.predict_embedding(record["center_embedding"], mode="prototype")
                else:
                    cls_result = cls_multi.predict_embedding(record["cls_embedding"])
                    center_result = center_multi.predict_embedding(record["center_embedding"])

                fused_scores = {
                    c: 0.5 * cls_result["class_scores"][c]
                    + 0.5 * center_result["class_scores"][c]
                    for c in classes
                }
                fused = rank_scores(fused_scores)
                pred = fused["predicted_class"]
                correct[mode][true] += int(pred == true)
                result_rows.append(
                    {
                        "shots": shots,
                        "mode": mode,
                        "true_class": true,
                        "image_path": str(record["image_path"]),
                        "predicted_class": pred,
                        "correct": int(pred == true),
                        "top1_similarity": f"{fused['top1_similarity']:.6f}",
                        "margin": f"{fused['margin']:.6f}",
                    }
                )

        total = sum(totals.values())
        for mode in MODES:
            total_correct = sum(correct[mode].values())
            thread_total = totals.get("thread_side", 0)
            thread_correct = correct[mode].get("thread_side", 0)
            row = {
                "shots": shots,
                "mode": mode,
                "query_total": total,
                "overall_accuracy": total_correct / max(1, total),
                "thread_side_accuracy": thread_correct / max(1, thread_total),
            }
            for c in classes:
                row[f"{c}_accuracy"] = correct[mode][c] / max(1, totals[c])
            summary_rows.append(row)
            print(
                f"{mode:<18} overall={row['overall_accuracy']:.2%} "
                f"thread_side={row['thread_side_accuracy']:.2%}"
            )

    write_rows(out / "results.csv", result_rows)
    write_rows(out / "summary.csv", summary_rows)
    write_rows(out / "cluster_sizes.csv", cluster_rows)

    print("\n=============== Multi-Prototype Fixed-Query 汇总 ===============")
    print(f"shared query start index: {args.query_start} (0-based), total={len(query)}")
    for shots in shot_counts:
        print(f"shots={shots}")
        for row in [r for r in summary_rows if r["shots"] == shots]:
            print(
                f"  {row['mode']:<18} overall={row['overall_accuracy']:.2%} "
                f"thread_side={row['thread_side_accuracy']:.2%}"
            )
    print(f"summary CSV: {(out/'summary.csv').resolve()}")
    print(f"cluster sizes: {(out/'cluster_sizes.csv').resolve()}")
    print("=============================================================")


if __name__ == "__main__":
    main()
