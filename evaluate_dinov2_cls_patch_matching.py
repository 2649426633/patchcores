from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from app.defect.defect_bank import DefectExemplarBank
from app.defect.patch_exemplar_matcher import (
    PatchExemplar,
    PatchExemplarMatcher,
    select_anomaly_patch_tokens,
)
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate anomaly-guided DINOv2 local patch exemplar matching and "
            "equal-weight CLS + patch-score fusion on identical PatchCore ROIs."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument("--patchcore-model-dir", default="products/screw/models/patchcore_320_l23")
    p.add_argument("--output-dir", default="outputs/screw/dinov2_cls_patch_matching")
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument(
        "--top-fraction",
        type=float,
        default=0.10,
        help="Fraction of hottest anomaly-guided DINO patch tokens retained. Fixed default=10%%.",
    )
    return p.parse_args()


def image_files(folder: Path):
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


def write_rows(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top-fraction must be in (0,1]")

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
    if pipeline.dinov2 is None:
        raise RuntimeError("DINOv2 failed to load")

    records = []
    print("\n========== Precompute common PatchCore ROI + DINO features ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(f"{class_dir.name}: need more than {args.shots} images")

        for index, image_path in enumerate(images):
            roi_result = pipeline.extract_roi(image_path)
            cls_embedding = pipeline.embed_roi(roi_result["roi"], feature_mode="cls")
            all_tokens = pipeline.dinov2.patch_tokens(roi_result["roi"])
            selected_tokens, selected_weights, selected_indices = select_anomaly_patch_tokens(
                all_tokens,
                roi_result["anomaly_roi"],
                top_fraction=args.top_fraction,
            )

            split = "support" if index < args.shots else "query"
            roi_path = out / "rois" / split / class_dir.name / image_path.name
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi_result["roi"].save(roi_path)

            anomaly_path = out / "anomaly_rois" / split / class_dir.name / image_path.with_suffix(".png").name
            anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            anomaly_img = np.clip(roi_result["anomaly_roi"] * 255.0, 0, 255).astype(np.uint8)
            from PIL import Image
            Image.fromarray(anomaly_img).save(anomaly_path)

            records.append(
                {
                    "split": split,
                    "true_class": class_dir.name,
                    "image_path": image_path,
                    "cls_embedding": cls_embedding,
                    "tokens": selected_tokens,
                    "weights": selected_weights,
                    "selected_indices": selected_indices,
                    "anomaly_score": roi_result["anomaly_score"],
                }
            )
            print(
                f"{split.upper():7} {class_dir.name}/{image_path.name} "
                f"patches={len(selected_tokens):02d} score={roi_result['anomaly_score']:.3f}"
            )

    support = [r for r in records if r["split"] == "support"]
    query = [r for r in records if r["split"] == "query"]

    labels = [r["true_class"] for r in support]
    paths = [str(r["image_path"].resolve()) for r in support]
    cls_bank = DefectExemplarBank(
        np.stack([r["cls_embedding"] for r in support], axis=0), labels, paths
    )
    patch_matcher = PatchExemplarMatcher(
        PatchExemplar(
            label=r["true_class"],
            image_path=str(r["image_path"].resolve()),
            tokens=r["tokens"],
            weights=r["weights"],
        )
        for r in support
    )

    if cls_bank.classes != patch_matcher.classes:
        raise RuntimeError("CLS and patch matcher classes differ")
    classes = cls_bank.classes

    rows = []
    totals = Counter()
    correct_patch = Counter()
    correct_fusion = Counter()
    confusion_patch = defaultdict(Counter)
    confusion_fusion = defaultdict(Counter)

    print("\n========== Evaluate local patch matching ==========")
    for i, r in enumerate(query, 1):
        cls_result = cls_bank.predict_embedding(r["cls_embedding"])
        patch_result = patch_matcher.predict(r["tokens"], r["weights"])

        fused_scores = {
            c: 0.5 * cls_result["class_scores"][c]
            + 0.5 * patch_result["class_scores"][c]
            for c in classes
        }
        fused = rank_scores(fused_scores)

        true = r["true_class"]
        patch_pred = patch_result["predicted_class"]
        fusion_pred = fused["predicted_class"]
        totals[true] += 1
        correct_patch[true] += int(patch_pred == true)
        correct_fusion[true] += int(fusion_pred == true)
        confusion_patch[true][patch_pred] += 1
        confusion_fusion[true][fusion_pred] += 1

        rows.append(
            {
                "true_class": true,
                "image_path": str(r["image_path"]),
                "selected_patch_count": len(r["tokens"]),
                "patch_pred": patch_pred,
                "patch_top1": f"{patch_result['top1_similarity']:.6f}",
                "patch_margin": f"{patch_result['margin']:.6f}",
                "patch_nearest_exemplar": patch_result["nearest_exemplar"],
                "cls_pred": cls_result["predicted_class"],
                "cls_top1": f"{cls_result['top1_similarity']:.6f}",
                "fusion_pred": fusion_pred,
                "fusion_top1": f"{fused['top1_similarity']:.6f}",
                "fusion_margin": f"{fused['margin']:.6f}",
                "patch_correct": int(patch_pred == true),
                "fusion_correct": int(fusion_pred == true),
            }
        )

        print(
            f"[{i:03d}/{len(query):03d}] {true}/{r['image_path'].name} "
            f"patch={patch_pred} fusion={fusion_pred} "
            f"{'OK' if fusion_pred == true else 'MISS'}"
        )

    write_rows(out / "fewshot_results.csv", rows)

    with open(out / "class_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["class", "total", "patch_correct", "patch_accuracy", "fusion_correct", "fusion_accuracy"])
        for c in classes:
            n = totals[c]
            w.writerow([
                c,
                n,
                correct_patch[c],
                f"{correct_patch[c] / max(1,n):.6f}",
                correct_fusion[c],
                f"{correct_fusion[c] / max(1,n):.6f}",
            ])

    with open(out / "confusion_matrix_fusion.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for t in classes:
            w.writerow([t, *[confusion_fusion[t][p] for p in classes]])

    total = sum(totals.values())
    patch_c = sum(correct_patch.values())
    fusion_c = sum(correct_fusion.values())

    print("\n=============== Patch-Level Exemplar Matching 汇总 ===============")
    print(f"Patch-only Top-1: {patch_c / max(1,total):.2%} ({patch_c}/{total})")
    print(f"CLS + PatchMatch 50/50 Top-1: {fusion_c / max(1,total):.2%} ({fusion_c}/{total})")
    print("----------------------------------------------------------------")
    for c in classes:
        n = totals[c]
        print(
            f"{c:<20} patch={correct_patch[c]:>2}/{n:<2}={correct_patch[c]/max(1,n):>7.2%} "
            f"fusion={correct_fusion[c]:>2}/{n:<2}={correct_fusion[c]/max(1,n):>7.2%}"
        )
    print(f"results: {(out/'fewshot_results.csv').resolve()}")
    print("================================================================")


if __name__ == "__main__":
    main()
