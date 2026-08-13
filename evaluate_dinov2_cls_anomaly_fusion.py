from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate 50/50 fusion of DINOv2 CLS and PatchCore-anomaly-weighted "
            "patch-token features using the same PatchCore ROIs and 3-shot split."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument("--patchcore-model-dir", default="products/screw/models/patchcore_320_l23")
    p.add_argument("--output-dir", default="outputs/screw/dinov2_cls_anomaly_fusion")
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    return p.parse_args()


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

    class_dirs = sorted(
        [p for p in test_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=args.patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
    )
    pipeline.load()

    support, query = [], []
    print("\n========== Build common PatchCore ROI + anomaly ROI ==========")
    for class_dir in class_dirs:
        files = image_files(class_dir)
        if len(files) <= args.shots:
            raise RuntimeError(f"{class_dir.name}: need more than shots={args.shots}")

        for idx, image_path in enumerate(files):
            result = pipeline.extract_roi(image_path)
            split = "support" if idx < args.shots else "query"

            roi_path = output_dir / "rois" / split / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            result["roi"].save(roi_path)

            weight_path = output_dir / "anomaly_rois" / split / class_dir.name / f"{image_path.stem}.png"
            weight_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(weight_path), np.clip(result["anomaly_roi"] * 255.0, 0, 255).astype(np.uint8))

            record = {
                "true_class": class_dir.name,
                "image_path": image_path,
                "roi": result["roi"],
                "anomaly_roi": result["anomaly_roi"],
                "anomaly_score": result["anomaly_score"],
                "bbox": result["bbox"],
            }
            (support if split == "support" else query).append(record)

            print(
                f"{split.upper():7} {class_dir.name}/{image_path.name} "
                f"score={result['anomaly_score']:.3f} bbox={result['bbox']}"
            )

    cls_support, weighted_support, labels, paths = [], [], [], []
    print("\n========== Build CLS + anomaly-weighted banks ==========")
    for i, r in enumerate(support, 1):
        cls_support.append(pipeline.embed_roi(r["roi"], feature_mode="cls"))
        weighted_support.append(
            pipeline.embed_roi(
                r["roi"],
                feature_mode="patch_weighted",
                spatial_weights=r["anomaly_roi"],
            )
        )
        labels.append(r["true_class"])
        paths.append(str(r["image_path"].resolve()))
        print(f"[{i:02d}/{len(support):02d}] {r['true_class']}/{r['image_path'].name}")

    cls_bank = DefectExemplarBank(np.stack(cls_support), labels, paths)
    weighted_bank = DefectExemplarBank(np.stack(weighted_support), labels, paths)
    cls_bank.save(output_dir / "bank_cls")
    weighted_bank.save(output_dir / "bank_patch_weighted")
    classes = cls_bank.classes

    rows = []
    total = Counter()
    correct = Counter()
    confusion = defaultdict(Counter)

    print("\n========== Evaluate CLS + anomaly-weighted 50/50 ==========")
    for i, r in enumerate(query, 1):
        cls_emb = pipeline.embed_roi(r["roi"], feature_mode="cls")
        weighted_emb = pipeline.embed_roi(
            r["roi"],
            feature_mode="patch_weighted",
            spatial_weights=r["anomaly_roi"],
        )
        cls_result = cls_bank.predict_embedding(cls_emb)
        weighted_result = weighted_bank.predict_embedding(weighted_emb)

        fused = {
            name: 0.5 * cls_result["class_scores"][name]
            + 0.5 * weighted_result["class_scores"][name]
            for name in classes
        }
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        pred, top1 = ranked[0]
        top2_class, top2 = ranked[1]
        margin = float(top1 - top2)

        true = r["true_class"]
        ok = pred == true
        total[true] += 1
        correct[true] += int(ok)
        confusion[true][pred] += 1

        rows.append(
            {
                "true_class": true,
                "predicted_class": pred,
                "correct": int(ok),
                "image_path": str(r["image_path"]),
                "anomaly_score": f"{r['anomaly_score']:.6f}",
                "bbox": ",".join(str(v) for v in r["bbox"]),
                "fused_top1_similarity": f"{top1:.6f}",
                "fused_top2_class": top2_class,
                "fused_top2_similarity": f"{top2:.6f}",
                "fused_margin": f"{margin:.6f}",
                "cls_pred": cls_result["predicted_class"],
                "weighted_pred": weighted_result["predicted_class"],
            }
        )
        print(
            f"[{i:03d}/{len(query):03d}] {true}/{r['image_path'].name} => {pred} "
            f"fused={top1:.4f} margin={margin:.4f} {'OK' if ok else 'MISS'}"
        )

    results_csv = output_dir / "fewshot_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    summary_csv = output_dir / "class_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["class", "correct", "total", "accuracy"])
        for name in classes:
            w.writerow([name, correct[name], total[name], f"{correct[name] / max(1,total[name]):.6f}"])

    confusion_csv = output_dir / "confusion_matrix.csv"
    with open(confusion_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["true\\pred", *classes])
        for true in classes:
            w.writerow([true, *[confusion[true][pred] for pred in classes]])

    n = sum(total.values()); c = sum(correct.values())
    print("\n=============== CLS + Anomaly-Weighted 50/50 汇总 ===============")
    print(f"Top-1 accuracy: {c/max(1,n):.2%} ({c}/{n})")
    for name in classes:
        print(f"{name:<20} {correct[name]:>3}/{total[name]:<3} = {correct[name]/max(1,total[name]):.2%}")
    print(f"results: {results_csv.resolve()}")
    print(f"class summary: {summary_csv.resolve()}")
    print(f"confusion matrix: {confusion_csv.resolve()}")
    print("================================================================")


if __name__ == "__main__":
    main()
