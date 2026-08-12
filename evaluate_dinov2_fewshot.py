from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import DINOv2Adapter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate frozen DINOv2 few-shot defect classification")
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--bank-dir", default="products/screw/defects/bank_3shot")
    parser.add_argument("--output-dir", default="outputs/screw/dinov2_3shot")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    return parser.parse_args()


def iter_queries(test_dir: Path, support_paths: set[str]):
    for class_dir in sorted(test_dir.iterdir(), key=lambda p: p.name.lower()):
        if not class_dir.is_dir() or class_dir.name.lower() == "good":
            continue
        for path in sorted(class_dir.iterdir(), key=lambda p: p.name.lower()):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
                continue
            if str(path.resolve()) in support_paths:
                continue
            yield class_dir.name, path


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    bank_dir = Path(args.bank_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bank = DefectExemplarBank.load(bank_dir)
    support_paths = {str(Path(p).resolve()) for p in bank.image_paths}

    extractor = DINOv2Adapter(device=args.device)
    extractor.load()

    queries = list(iter_queries(test_dir, support_paths))
    if not queries:
        raise RuntimeError("No query images remain after excluding support images")

    rows = []
    per_class_total = Counter()
    per_class_correct = Counter()
    confusion = defaultdict(Counter)

    print("\n========== DINOv2 3-shot evaluation ==========")
    print(f"bank classes: {bank.classes}")
    print(f"support exemplars: {len(bank.labels)}")
    print(f"query images: {len(queries)}")
    print("================================================")

    for index, (true_class, image_path) in enumerate(queries, start=1):
        embedding = extractor.embed(image_path)
        result = bank.predict_embedding(embedding)
        pred = result["predicted_class"]
        correct = pred == true_class

        per_class_total[true_class] += 1
        per_class_correct[true_class] += int(correct)
        confusion[true_class][pred] += 1

        row = {
            "true_class": true_class,
            "predicted_class": pred,
            "correct": int(correct),
            "image_path": str(image_path),
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
            f"=> {pred}  sim={result['top1_similarity']:.4f} "
            f"margin={result['margin']:.4f}  {mark}"
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
            writer.writerow([true_class, *[confusion[true_class][pred] for pred in classes]])

    total = sum(per_class_total.values())
    correct = sum(per_class_correct.values())
    accuracy = correct / total

    summary_csv = output_dir / "class_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "correct", "total", "accuracy"])
        for class_name in classes:
            n = per_class_total[class_name]
            c = per_class_correct[class_name]
            writer.writerow([class_name, c, n, f"{(c / n if n else 0.0):.6f}"])

    print("\n=============== 3-shot 汇总 ===============")
    print(f"Top-1 accuracy: {accuracy:.2%} ({correct}/{total})")
    for class_name in classes:
        n = per_class_total[class_name]
        c = per_class_correct[class_name]
        print(f"{class_name:<20} {c:>3}/{n:<3} = {(c / n if n else 0.0):.2%}")
    print(f"results: {results_csv.resolve()}")
    print(f"class summary: {summary_csv.resolve()}")
    print(f"confusion matrix: {confusion_csv.resolve()}")
    print("===========================================")


if __name__ == "__main__":
    main()
