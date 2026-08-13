from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dynamic_feature_selector import (
    DynamicCLSPatchSelector,
    SelectorCalibration,
)
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
            "Evaluate a support-only dynamic selector between DINOv2 CLS and "
            "anomaly-guided PatchMatch. Query labels are evaluation-only."
        )
    )
    p.add_argument("--test-dir", default="data/screw/test")
    p.add_argument(
        "--patchcore-model-dir",
        default="products/screw/models/patchcore_320_l23",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/screw/dinov2_dynamic_selector",
    )
    p.add_argument("--shots", type=int, default=3)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--top-fraction", type=float, default=0.10)
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
        "class_scores": scores,
    }


def make_cls_bank(records: list[dict]) -> DefectExemplarBank:
    return DefectExemplarBank(
        np.stack([r["cls_embedding"] for r in records], axis=0),
        [r["true_class"] for r in records],
        [str(r["image_path"].resolve()) for r in records],
    )


def make_patch_matcher(records: list[dict]) -> PatchExemplarMatcher:
    return PatchExemplarMatcher(
        PatchExemplar(
            label=r["true_class"],
            image_path=str(r["image_path"].resolve()),
            tokens=r["tokens"],
            weights=r["weights"],
        )
        for r in records
    )


def write_confusion(path: Path, classes: list[str], confusion) -> None:
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["true\\pred", *classes])
        for true_class in classes:
            w.writerow([true_class, *[confusion[true_class][p] for p in classes]])


def main():
    args = parse_args()
    test_dir = Path(args.test_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.shots < 2:
        raise ValueError("--shots must be >= 2 for support leave-one-out calibration")
    if not (0.0 < args.top_fraction <= 1.0):
        raise ValueError("--top-fraction must be in (0,1]")
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
    if pipeline.dinov2 is None:
        raise RuntimeError("DINOv2 failed to load")

    records: list[dict] = []
    print("\n========== Precompute common PatchCore ROI + CLS + Patch tokens ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) <= args.shots:
            raise RuntimeError(
                f"{class_dir.name}: need more than shots={args.shots} images"
            )

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

            anomaly_path = (
                out
                / "anomaly_rois"
                / split
                / class_dir.name
                / image_path.with_suffix(".png").name
            )
            anomaly_path.parent.mkdir(parents=True, exist_ok=True)
            anomaly_img = np.clip(
                roi_result["anomaly_roi"] * 255.0, 0, 255
            ).astype(np.uint8)
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

    print("\n========== Support leave-one-out selector calibration ==========")
    loo_records: list[dict] = []
    for index, held in enumerate(support):
        remaining = [r for j, r in enumerate(support) if j != index]
        cls_result = make_cls_bank(remaining).predict_embedding(held["cls_embedding"])
        patch_result = make_patch_matcher(remaining).predict(
            held["tokens"], held["weights"]
        )
        loo_records.append(
            {
                "support_index": index,
                "true_class": held["true_class"],
                "image_path": str(held["image_path"]),
                "cls_pred": cls_result["predicted_class"],
                "cls_top1": float(cls_result["top1_similarity"]),
                "cls_margin": float(cls_result["margin"]),
                "cls_correct": int(cls_result["predicted_class"] == held["true_class"]),
                "patch_pred": patch_result["predicted_class"],
                "patch_top1": float(patch_result["top1_similarity"]),
                "patch_margin": float(patch_result["margin"]),
                "patch_correct": int(
                    patch_result["predicted_class"] == held["true_class"]
                ),
            }
        )
        print(
            f"[{index+1:02d}/{len(support):02d}] {held['true_class']}/{held['image_path'].name} "
            f"CLS={cls_result['predicted_class']} "
            f"Patch={patch_result['predicted_class']}"
        )

    calibration = SelectorCalibration.from_loo_records(loo_records)
    selector = DynamicCLSPatchSelector(calibration)

    for row in loo_records:
        cls_stub = {
            "top1_similarity": row["cls_top1"],
            "margin": row["cls_margin"],
        }
        patch_stub = {
            "top1_similarity": row["patch_top1"],
            "margin": row["patch_margin"],
        }
        cls_conf = calibration.method_confidence(cls_stub, "cls")
        patch_conf = calibration.method_confidence(patch_stub, "patch")
        row["cls_confidence_percentile"] = cls_conf["confidence"]
        row["patch_confidence_percentile"] = patch_conf["confidence"]

    write_rows(out / "support_loo_calibration.csv", loo_records)
    with open(out / "selector_calibration.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "calibration_source": "support_leave_one_out_only",
                "support_count": calibration.support_count,
                "cls_loo_accuracy": calibration.cls_loo_accuracy,
                "patch_loo_accuracy": calibration.patch_loo_accuracy,
                "confidence": "mean(empirical_percentile(top1_similarity), empirical_percentile(margin))",
                "disagreement_rule": "choose_higher_calibrated_confidence",
                "tie_break": "higher_support_loo_accuracy_then_cls",
                "top_fraction": args.top_fraction,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    cls_bank = make_cls_bank(support)
    patch_matcher = make_patch_matcher(support)
    if cls_bank.classes != patch_matcher.classes:
        raise RuntimeError("CLS and PatchMatch classes differ")
    classes = cls_bank.classes

    totals = Counter()
    correct = {
        "cls": Counter(),
        "patch": Counter(),
        "fixed_fusion": Counter(),
        "dynamic": Counter(),
        "oracle_cls_or_patch": Counter(),
    }
    confusion = {
        "cls": defaultdict(Counter),
        "patch": defaultdict(Counter),
        "fixed_fusion": defaultdict(Counter),
        "dynamic": defaultdict(Counter),
    }
    selector_sources = Counter()
    disagreement_cases = Counter()
    rows: list[dict] = []

    print("\n========== Evaluate dynamic CLS / PatchMatch selector ==========")
    for i, r in enumerate(query, 1):
        cls_result = cls_bank.predict_embedding(r["cls_embedding"])
        patch_result = patch_matcher.predict(r["tokens"], r["weights"])

        fixed_scores = {
            c: 0.5 * cls_result["class_scores"][c]
            + 0.5 * patch_result["class_scores"][c]
            for c in classes
        }
        fixed_result = rank_scores(fixed_scores)
        dynamic_result = selector.select(cls_result, patch_result)

        true = r["true_class"]
        cls_pred = cls_result["predicted_class"]
        patch_pred = patch_result["predicted_class"]
        fixed_pred = fixed_result["predicted_class"]
        dynamic_pred = dynamic_result["predicted_class"]

        cls_ok = cls_pred == true
        patch_ok = patch_pred == true
        fixed_ok = fixed_pred == true
        dynamic_ok = dynamic_pred == true
        oracle_ok = cls_ok or patch_ok

        totals[true] += 1
        correct["cls"][true] += int(cls_ok)
        correct["patch"][true] += int(patch_ok)
        correct["fixed_fusion"][true] += int(fixed_ok)
        correct["dynamic"][true] += int(dynamic_ok)
        correct["oracle_cls_or_patch"][true] += int(oracle_ok)

        confusion["cls"][true][cls_pred] += 1
        confusion["patch"][true][patch_pred] += 1
        confusion["fixed_fusion"][true][fixed_pred] += 1
        confusion["dynamic"][true][dynamic_pred] += 1
        selector_sources[dynamic_result["selected_method"]] += 1

        if cls_pred != patch_pred:
            if cls_ok and not patch_ok:
                disagreement_cases["only_cls_correct"] += 1
            elif patch_ok and not cls_ok:
                disagreement_cases["only_patch_correct"] += 1
            else:
                disagreement_cases["neither_correct"] += 1
            disagreement_cases[
                "dynamic_correct" if dynamic_ok else "dynamic_wrong"
            ] += 1
        else:
            disagreement_cases["agree"] += 1
            disagreement_cases[
                "agree_correct" if cls_ok else "agree_wrong"
            ] += 1

        rows.append(
            {
                "true_class": true,
                "image_path": str(r["image_path"]),
                "selected_patch_count": len(r["tokens"]),
                "cls_pred": cls_pred,
                "cls_top1": f"{cls_result['top1_similarity']:.6f}",
                "cls_margin": f"{cls_result['margin']:.6f}",
                "cls_confidence": f"{dynamic_result['cls_confidence']:.6f}",
                "patch_pred": patch_pred,
                "patch_top1": f"{patch_result['top1_similarity']:.6f}",
                "patch_margin": f"{patch_result['margin']:.6f}",
                "patch_confidence": f"{dynamic_result['patch_confidence']:.6f}",
                "patch_nearest_exemplar": patch_result["nearest_exemplar"],
                "agree": int(dynamic_result["agreed"]),
                "selected_method": dynamic_result["selected_method"],
                "confidence_delta_cls_minus_patch": f"{dynamic_result['confidence_delta_cls_minus_patch']:.6f}",
                "fixed_fusion_pred": fixed_pred,
                "dynamic_pred": dynamic_pred,
                "cls_correct": int(cls_ok),
                "patch_correct": int(patch_ok),
                "fixed_fusion_correct": int(fixed_ok),
                "dynamic_correct": int(dynamic_ok),
                "oracle_cls_or_patch_correct": int(oracle_ok),
            }
        )

        print(
            f"[{i:03d}/{len(query):03d}] {true}/{r['image_path'].name} "
            f"CLS={cls_pred} Patch={patch_pred} "
            f"Dynamic={dynamic_pred} via={dynamic_result['selected_method']} "
            f"{'OK' if dynamic_ok else 'MISS'}"
        )

    write_rows(out / "fewshot_results.csv", rows)

    summary_rows = []
    for c in classes:
        n = totals[c]
        summary_rows.append(
            {
                "class": c,
                "total": n,
                "cls_correct": correct["cls"][c],
                "cls_accuracy": correct["cls"][c] / max(1, n),
                "patch_correct": correct["patch"][c],
                "patch_accuracy": correct["patch"][c] / max(1, n),
                "fixed_fusion_correct": correct["fixed_fusion"][c],
                "fixed_fusion_accuracy": correct["fixed_fusion"][c] / max(1, n),
                "dynamic_correct": correct["dynamic"][c],
                "dynamic_accuracy": correct["dynamic"][c] / max(1, n),
                "oracle_cls_or_patch_correct": correct["oracle_cls_or_patch"][c],
                "oracle_cls_or_patch_accuracy": correct["oracle_cls_or_patch"][c]
                / max(1, n),
            }
        )
    write_rows(out / "class_summary.csv", summary_rows)

    for method in ["cls", "patch", "fixed_fusion", "dynamic"]:
        write_confusion(
            out / f"confusion_matrix_{method}.csv",
            classes,
            confusion[method],
        )

    thread_side_rows = []
    if "thread_side" in classes:
        for method in ["cls", "patch", "fixed_fusion", "dynamic"]:
            for pred in classes:
                count = confusion[method]["thread_side"][pred]
                if count:
                    thread_side_rows.append(
                        {
                            "method": method,
                            "true_class": "thread_side",
                            "predicted_class": pred,
                            "count": count,
                        }
                    )
    write_rows(out / "thread_side_confusion.csv", thread_side_rows)

    write_rows(
        out / "selector_source_summary.csv",
        [{"source": key, "count": value} for key, value in selector_sources.items()],
    )
    write_rows(
        out / "disagreement_summary.csv",
        [{"case": key, "count": value} for key, value in disagreement_cases.items()],
    )

    total = sum(totals.values())
    method_totals = {
        method: sum(correct[method].values())
        for method in correct
    }

    print("\n=============== Dynamic CLS / PatchMatch Selector 汇总 ===============")
    print(
        f"Support LOO CLS accuracy:   {calibration.cls_loo_accuracy:.2%}"
    )
    print(
        f"Support LOO Patch accuracy: {calibration.patch_loo_accuracy:.2%}"
    )
    print("--------------------------------------------------------------------")
    print(
        f"CLS-only:                  {method_totals['cls']/max(1,total):.2%} "
        f"({method_totals['cls']}/{total})"
    )
    print(
        f"Patch-only:                {method_totals['patch']/max(1,total):.2%} "
        f"({method_totals['patch']}/{total})"
    )
    print(
        f"CLS + Patch 50/50:         {method_totals['fixed_fusion']/max(1,total):.2%} "
        f"({method_totals['fixed_fusion']}/{total})"
    )
    print(
        f"Dynamic selector:          {method_totals['dynamic']/max(1,total):.2%} "
        f"({method_totals['dynamic']}/{total})"
    )
    print(
        f"Oracle CLS-or-Patch upper: {method_totals['oracle_cls_or_patch']/max(1,total):.2%} "
        f"({method_totals['oracle_cls_or_patch']}/{total}) [diagnostic only]"
    )
    print("--------------------------------------------------------------------")
    for c in classes:
        n = totals[c]
        print(
            f"{c:<20} "
            f"CLS={correct['cls'][c]:>2}/{n:<2}={correct['cls'][c]/max(1,n):>7.2%} "
            f"Patch={correct['patch'][c]:>2}/{n:<2}={correct['patch'][c]/max(1,n):>7.2%} "
            f"Fixed={correct['fixed_fusion'][c]:>2}/{n:<2}={correct['fixed_fusion'][c]/max(1,n):>7.2%} "
            f"Dynamic={correct['dynamic'][c]:>2}/{n:<2}={correct['dynamic'][c]/max(1,n):>7.2%} "
            f"Oracle={correct['oracle_cls_or_patch'][c]:>2}/{n:<2}={correct['oracle_cls_or_patch'][c]/max(1,n):>7.2%}"
        )

    if "thread_side" in classes:
        print("--------------------------------------------------------------------")
        print("thread_side dynamic confusion:")
        for pred in classes:
            count = confusion["dynamic"]["thread_side"][pred]
            if count:
                print(f"  thread_side -> {pred:<20} {count}")

    disagreements = (
        disagreement_cases["only_cls_correct"]
        + disagreement_cases["only_patch_correct"]
        + disagreement_cases["neither_correct"]
    )
    print("--------------------------------------------------------------------")
    print(f"agree cases: {disagreement_cases['agree']}/{total}")
    print(f"disagreement cases: {disagreements}/{total}")
    if disagreements:
        print(
            "dynamic correct on disagreements: "
            f"{disagreement_cases['dynamic_correct']}/{disagreements} "
            f"= {disagreement_cases['dynamic_correct']/disagreements:.2%}"
        )
        print(
            f"only CLS correct:   {disagreement_cases['only_cls_correct']}"
        )
        print(
            f"only Patch correct: {disagreement_cases['only_patch_correct']}"
        )
        print(
            f"neither correct:    {disagreement_cases['neither_correct']}"
        )
    print(f"results: {(out/'fewshot_results.csv').resolve()}")
    print(f"support calibration: {(out/'support_loo_calibration.csv').resolve()}")
    print(f"thread-side diagnosis: {(out/'thread_side_confusion.csv').resolve()}")
    print("====================================================================")


if __name__ == "__main__":
    main()
