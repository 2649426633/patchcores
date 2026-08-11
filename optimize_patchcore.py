from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run controlled PatchCore localization experiments and select the best "
            "candidate without overwriting the baseline model."
        )
    )
    parser.add_argument("--normal-dir", default="data/screw/train/good")
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--ground-truth-dir", default="data/screw/ground_truth")
    parser.add_argument(
        "--baseline-model-dir",
        default="products/screw/models/patchcore",
    )
    parser.add_argument(
        "--best-model-dir",
        default="products/screw/models/patchcore_best",
    )
    parser.add_argument(
        "--work-dir",
        default="outputs/screw/patchcore_experiments",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    print("\n$ " + " ".join(command))
    env = os.environ.copy()
    # Child-process-only workaround for the known Windows OpenMP conflict.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    completed = subprocess.run(command, cwd=ROOT, env=env)
    return int(completed.returncode)


def model_is_complete(model_dir: Path) -> bool:
    return (
        (model_dir / "patchcore_params.pkl").is_file()
        and (model_dir / "nnscorer_search_index.faiss").is_file()
    )


def evaluate_model(
    model_dir: Path,
    output_dir: Path,
    args,
) -> Path | None:
    command = [
        sys.executable,
        "evaluate_mvtec_screw.py",
        "--test-dir",
        str(args.test_dir),
        "--ground-truth-dir",
        str(args.ground_truth_dir),
        "--model-dir",
        str(model_dir),
        "--output-dir",
        str(output_dir),
        "--bbox-relative-threshold",
        str(args.bbox_relative_threshold),
    ]
    if args.device:
        command.extend(["--device", args.device])

    if run_command(command) != 0:
        return None

    csv_path = output_dir / "localization_results.csv"
    return csv_path if csv_path.exists() else None


def train_model(
    model_dir: Path,
    imagesize: int,
    layers: tuple[str, ...],
    args,
) -> bool:
    command = [
        sys.executable,
        "train_patchcore.py",
        "--normal-dir",
        str(args.normal_dir),
        "--model-dir",
        str(model_dir),
        "--batch-size",
        str(args.batch_size),
        "--coreset",
        str(args.coreset),
        "--imagesize",
        str(imagesize),
        "--layers",
        *layers,
    ]
    if args.device:
        command.extend(["--device", args.device])
    return run_command(command) == 0


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def summarize(csv_path: Path) -> dict:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            try:
                row["_iou"] = float(row["bbox_iou"])
            except (TypeError, ValueError):
                continue
            row["_peak"] = parse_bool(row.get("peak_in_gt_bbox", ""))
            rows.append(row)

    if not rows:
        raise RuntimeError(f"No valid evaluation rows in {csv_path}")

    def metrics(subset):
        if not subset:
            return {
                "count": 0,
                "mean_iou": 0.0,
                "median_iou": 0.0,
                "iou20": 0.0,
                "iou50": 0.0,
                "peak_gt": 0.0,
            }
        ious = [r["_iou"] for r in subset]
        return {
            "count": len(subset),
            "mean_iou": sum(ious) / len(ious),
            "median_iou": statistics.median(ious),
            "iou20": sum(v >= 0.20 for v in ious) / len(ious),
            "iou50": sum(v >= 0.50 for v in ious) / len(ious),
            "peak_gt": sum(r["_peak"] for r in subset) / len(subset),
        }

    result = {"overall": metrics(rows)}
    for defect_type in ["manipulated_front", "scratch_head", "scratch_neck", "thread_side", "thread_top"]:
        subset = [r for r in rows if r.get("defect_type") == defect_type]
        result[defect_type] = metrics(subset)
    return result


def flatten_summary(name: str, model_dir: Path, summary: dict) -> dict:
    overall = summary["overall"]
    thread_side = summary["thread_side"]
    scratch_head = summary["scratch_head"]
    return {
        "experiment": name,
        "model_dir": str(model_dir),
        "overall_mean_iou": overall["mean_iou"],
        "overall_median_iou": overall["median_iou"],
        "overall_iou20": overall["iou20"],
        "overall_iou50": overall["iou50"],
        "overall_peak_gt": overall["peak_gt"],
        "thread_side_mean_iou": thread_side["mean_iou"],
        "thread_side_iou50": thread_side["iou50"],
        "thread_side_peak_gt": thread_side["peak_gt"],
        "scratch_head_mean_iou": scratch_head["mean_iou"],
        "scratch_head_iou50": scratch_head["iou50"],
        "scratch_head_peak_gt": scratch_head["peak_gt"],
    }


def print_summary(row: dict):
    print(
        f"{row['experiment']:<20} "
        f"overall_peak={row['overall_peak_gt']*100:6.2f}%  "
        f"meanIoU={row['overall_mean_iou']:.4f}  "
        f"IoU>=0.50={row['overall_iou50']*100:6.2f}%  "
        f"thread_side_peak={row['thread_side_peak_gt']*100:6.2f}%  "
        f"scratch_head_peak={row['scratch_head_peak_gt']*100:6.2f}%"
    )


def main():
    args = parse_args()
    args.normal_dir = Path(args.normal_dir)
    args.test_dir = Path(args.test_dir)
    args.ground_truth_dir = Path(args.ground_truth_dir)
    args.baseline_model_dir = Path(args.baseline_model_dir)
    args.best_model_dir = Path(args.best_model_dir)
    args.work_dir = Path(args.work_dir)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    experiments = [
        {
            "name": "baseline_224_l23",
            "model_dir": args.baseline_model_dir,
            "train": False,
            "imagesize": 224,
            "layers": ("layer2", "layer3"),
        },
        {
            "name": "hires_320_l23",
            "model_dir": Path("products/screw/models/patchcore_320_l23"),
            "train": True,
            "imagesize": 320,
            "layers": ("layer2", "layer3"),
        },
        {
            "name": "texture_224_l12",
            "model_dir": Path("products/screw/models/patchcore_224_l12"),
            "train": True,
            "imagesize": 224,
            "layers": ("layer1", "layer2"),
        },
    ]

    summaries = []

    for experiment in experiments:
        name = experiment["name"]
        model_dir = experiment["model_dir"]
        eval_dir = args.work_dir / name

        print("\n" + "=" * 70)
        print(f"实验: {name}")
        print(f"模型: {model_dir}")
        print("=" * 70)

        if experiment["train"]:
            if model_is_complete(model_dir):
                print(f"已有完整实验模型，跳过训练: {model_dir}")
            else:
                if model_dir.exists():
                    print(f"发现不完整实验模型，删除后重训: {model_dir}")
                    shutil.rmtree(model_dir)
                ok = train_model(
                    model_dir=model_dir,
                    imagesize=experiment["imagesize"],
                    layers=experiment["layers"],
                    args=args,
                )
                if not ok or not model_is_complete(model_dir):
                    print(f"训练失败或模型文件不完整，跳过实验: {name}")
                    continue
        elif not model_is_complete(model_dir):
            print(f"基线模型不完整，跳过实验: {model_dir}")
            continue

        csv_path = evaluate_model(model_dir, eval_dir, args)
        if csv_path is None:
            print(f"评估失败，跳过实验: {name}")
            continue

        summary = summarize(csv_path)
        row = flatten_summary(name, model_dir, summary)
        summaries.append(row)
        print_summary(row)

    if not summaries:
        raise RuntimeError("No experiment completed successfully.")

    summary_csv = args.work_dir / "experiment_summary.csv"
    with open(summary_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    # Main optimization target: the current failure cluster is thread_side MAP
    # localization. Prefer higher thread-side peak hit rate, then overall peak hit
    # rate, then mean IoU. This keeps model selection tied to the measured failure.
    best = max(
        summaries,
        key=lambda r: (
            r["thread_side_peak_gt"],
            r["overall_peak_gt"],
            r["overall_mean_iou"],
        ),
    )

    best_source = Path(best["model_dir"])
    if args.best_model_dir.exists():
        shutil.rmtree(args.best_model_dir)
    shutil.copytree(best_source, args.best_model_dir)

    print("\n" + "=" * 70)
    print("实验汇总")
    print("=" * 70)
    for row in summaries:
        print_summary(row)

    print("\n最佳候选:")
    print(f"  {best['experiment']}")
    print(f"  thread_side peak命中率: {best['thread_side_peak_gt']*100:.2f}%")
    print(f"  overall peak命中率:     {best['overall_peak_gt']*100:.2f}%")
    print(f"  overall mean IoU:       {best['overall_mean_iou']:.4f}")
    print(f"  已复制到: {args.best_model_dir.resolve()}")
    print(f"  汇总CSV: {summary_csv.resolve()}")
    print("\n原 baseline 模型未被覆盖。")


if __name__ == "__main__":
    main()
