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
            "Evaluate the complete PatchCore ROI -> frozen DINOv2 -> exemplar bank "
            "pipeline on MVTec screw. GT masks are not used."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    parser.add_argument(
        "--bank-dir",
        default="products/screw/defects/bank_patchcore_roi_3shot",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/patchcore_dinov2_3shot",
    )
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--roi-margin", type=float, default=0.50)
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    bank_dir = Path(args.bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
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

    embeddings = []
    labels = []
    support_paths = []
    queries = []

    print("\n========== Build PatchCore-ROI 3-shot bank ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images; need more than shots={args.shots}"
            )

        for index, image_path in enumerate(images):
            if index < args.shots:
                roi_result = pipeline.extract_roi(image_path)
                embedding = pipeline.embed_roi(roi_result["roi"])

                roi_path = output_dir / "support_rois" / class_dir.name / image_path.name
                roi_path.parent.mkdir(parents=True, exist_ok=True)
                roi_result["roi"].save(roi_path)

                overlay_path = output_dir / "support_overlays" / class_dir.name / image_path.name
                pipeline.save_bbox_overlay(
                    roi_result["display_image"], roi_result["bbox"], overlay_path
                )

                embeddings.append(embedding)
                labels.append(class_dir.name)
                support_paths.append(str(image_path.resolve()))

                print(
                    f"SUPPORT {class_dir.name}/{image_path.name} "
                    f"score={roi_result['anomaly_score']:.3f} "
                    f"bbox={roi_result['bbox']} source={roi_result['bbox_source']}"
                )
            else:
                queries.append((class_dir.name, image_path))

    bank = DefectExemplarBank(
        np.stack(embeddings, axis=0),
        labels,
        support_paths,
    )
    bank.save(bank_dir)
    pipeline.bank = bank

    print("\n========== End-to-end PatchCore -> DINOv2 evaluation ==========")
    print(f"classes: {bank.classes}")
    print(f"support exemplars: {len(bank.labels)}")
    print(f"query images: {len(queries)}")
    print(f"bbox threshold: {args.bbox_relative_threshold:.2f}")
    print(f"ROI margin: {args.roi_margin:.2f}")
    print("===============================================================")

    rows = []
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)
    fallback_count = 0

    for index, (true_class, image_path) in enumerate(queries, start=1):
        result = pipeline.classify(image_path)
        pred = result["predicted_class"]
        correct = pred == true_class
        fallback_count += int(result["bbox_source"] == "peak_fallback")

        per_class_total[true_class] += 1
        per_class_correct[true_class] += int(correct)
        confusion[true_class][pred] += 1

        roi_path = output_dir / "query_rois" / true_class / image_path.name
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        result["roi"].save(roi_path)

        overlay_path = output_dir / "query_overlays" / true_class / image_path.name
        pipeline.save_bbox_overlay(
            result["display_image"], result["bbox"], overlay_path
        )

        row = {
            "true_class": true_class,
            "predicted_class": pred,
            "correct": int(correct),
            "image_path": str(image_path),
            "anomaly_score": f"{result['anomaly_score']:.6f}",
            "bbox": ",".join(str(v) for v in result["bbox"]),
            "expanded_bbox": ",".join(str(v) for v in result["expanded_bbox"]),
            "bbox_source": result["bbox_source"],
            "roi_path": str(roi_path.resolve()),
            "overlay_path": str(overlay_path.resolve()),
            "top1_similarity": f"{result['top1_similarity']:.6f}",
            "top2_class": result["top2_class"] or "",
            "top2_similarity": f"{result['top2_similarity']:.6f}",
            "margin": f"{result['margin']:.6f}",
            "nearest_exemplar": result["nearest_exemplar"],
        }
        rows.append(row)

        mark = "OK" if correct else "MISS"
        print(
            f"[{index:03d}/{len(queries):03d}] {true_class}/{image_path.name} "
            f"=> {pred}  patch={result['anomaly_score']:.3f} "
            f"sim={result['top1_similarity']:.4f} margin={result['margin']:.4f} "
            f"bbox={result['bbox_source']}  {mark}"
        )

    results_csv = output_dir / "fewshot_results.csv"
    with open(results_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    classes = sorted(set(bank.classes) | set(per_class_total.keys()))

    confusion_csv = output_dir / "confusion_matrix.csv"
    with open(confusion_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for true_class in classes:
            writer.writerow(
                [true_class, *[confusion[true_class][pred] for pred in classes]]
            )

    summary_csv = output_dir / "class_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for class_name in classes:
            n = per_class_total[class_name]
            c = per_class_correct[class_name]
            writer.writerow([class_name, c, n, f"{(c / n if n else 0.0):.6f}"])

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())

    print("\n========== PatchCore ROI -> DINOv2 3-shot 汇总 ==========")
    print(f"Top-1 accuracy: {correct / max(1, total):.2%} ({correct}/{total})")
    for class_name in classes:
        n = per_class_total[class_name]
        c = per_class_correct[class_name]
        print(f"{class_name:<20} {c:>3}/{n:<3} = {c / max(1, n):.2%}")
    print(f"peak fallback ROI: {fallback_count}/{total}")
    print(f"results: {results_csv.resolve()}")
    print(f"class summary: {summary_csv.resolve()}")
    print(f"confusion matrix: {confusion_csv.resolve()}")
    print("=========================================================")


if __name__ == "__main__":
    main()
