from __future__ import annotations

import argparse
import csv
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESOLUTION = 320
EXPERIMENTS = [
    ("l23", ("layer2", "layer3"), Path("products/screw/models/patchcore_320_l23")),
    ("l12", ("layer1", "layer2"), Path("products/screw/models/patchcore_320_l12")),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare PatchCore feature layers at fixed 320x320 input resolution. "
            "Only layers are changed; resolution, coreset and other PatchCore settings remain fixed."
        )
    )
    parser.add_argument("--normal-dir", default="data/screw/train/good")
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--ground-truth-dir", default="data/screw/ground_truth")
    parser.add_argument("--work-dir", default="outputs/screw/layer_comparison_320")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    print("\n$ " + " ".join(command))
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    result = subprocess.run(command, cwd=ROOT, env=env)
    return int(result.returncode)


def model_matches(model_dir: Path, layers: tuple[str, ...]) -> bool:
    params_file = model_dir / "patchcore_params.pkl"
    index_file = model_dir / "nnscorer_search_index.faiss"
    if not params_file.exists() or not index_file.exists():
        return False
    try:
        with open(params_file, "rb") as f:
            saved = pickle.load(f)
        input_shape = saved.get("input_shape")
        saved_layers = tuple(saved.get("layers_to_extract_from", ()))
        return (
            input_shape is not None
            and int(input_shape[-1]) == RESOLUTION
            and saved_layers == layers
        )
    except Exception:
        return False


def train_model(model_dir: Path, layers: tuple[str, ...], args) -> bool:
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
        str(RESOLUTION),
        "--layers",
        *layers,
    ]
    if args.device:
        command.extend(["--device", args.device])
    return run_command(command) == 0


def evaluate_model(model_dir: Path, output_dir: Path, args) -> Path | None:
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


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(csv_path: Path) -> list[dict]:
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
    return rows


def metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "count": 0,
            "peak_gt": 0.0,
            "mean_iou": 0.0,
            "median_iou": 0.0,
            "iou20": 0.0,
            "iou50": 0.0,
        }
    ious = [r["_iou"] for r in rows]
    return {
        "count": len(rows),
        "peak_gt": sum(r["_peak"] for r in rows) / len(rows),
        "mean_iou": sum(ious) / len(ious),
        "median_iou": statistics.median(ious),
        "iou20": sum(v >= 0.20 for v in ious) / len(ious),
        "iou50": sum(v >= 0.50 for v in ious) / len(ious),
    }


def summarize(name: str, layers: tuple[str, ...], model_dir: Path, csv_path: Path) -> dict:
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No valid rows in {csv_path}")

    overall = metrics(rows)
    by_type = {}
    for defect_type in [
        "manipulated_front",
        "scratch_head",
        "scratch_neck",
        "thread_side",
        "thread_top",
    ]:
        by_type[defect_type] = metrics(
            [r for r in rows if r.get("defect_type") == defect_type]
        )

    return {
        "experiment": name,
        "resolution": RESOLUTION,
        "layers": "+".join(layers),
        "model_dir": str(model_dir),
        "overall_peak_gt": overall["peak_gt"],
        "overall_mean_iou": overall["mean_iou"],
        "overall_median_iou": overall["median_iou"],
        "overall_iou20": overall["iou20"],
        "overall_iou50": overall["iou50"],
        "manipulated_front_peak_gt": by_type["manipulated_front"]["peak_gt"],
        "scratch_head_peak_gt": by_type["scratch_head"]["peak_gt"],
        "scratch_neck_peak_gt": by_type["scratch_neck"]["peak_gt"],
        "thread_side_peak_gt": by_type["thread_side"]["peak_gt"],
        "thread_top_peak_gt": by_type["thread_top"]["peak_gt"],
        "thread_side_mean_iou": by_type["thread_side"]["mean_iou"],
        "scratch_head_mean_iou": by_type["scratch_head"]["mean_iou"],
    }


def pct(v: float) -> str:
    return f"{100.0 * v:.2f}%"


def main():
    args = parse_args()
    args.normal_dir = Path(args.normal_dir)
    args.test_dir = Path(args.test_dir)
    args.ground_truth_dir = Path(args.ground_truth_dir)
    args.work_dir = Path(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    print("========== 320px 特征层单因素比较 ==========")
    print("固定 imagesize: 320")
    print(f"固定 coreset: {args.coreset}")
    print("A: layer2 + layer3")
    print("B: layer1 + layer2")
    print("除特征层外，其余PatchCore配置保持一致。")
    print("==========================================")

    summaries = []

    for name, layers, model_dir in EXPERIMENTS:
        print("\n" + "=" * 70)
        print(f"实验: 320_{name}  layers={layers}")
        print(f"模型: {model_dir}")
        print("=" * 70)

        matched = model_matches(model_dir, layers)
        if args.force_retrain or not matched:
            if model_dir.exists() and not matched:
                print(f"检测到不匹配/不完整模型，将重新训练并覆盖目录: {model_dir}")
            ok = train_model(model_dir, layers, args)
            if not ok:
                print(f"训练失败，跳过: {name}")
                continue
        else:
            print("已有匹配模型，跳过训练。")

        eval_dir = args.work_dir / f"eval_320_{name}"
        csv_path = evaluate_model(model_dir, eval_dir, args)
        if csv_path is None:
            print(f"评估失败，跳过: {name}")
            continue

        row = summarize(name, layers, model_dir, csv_path)
        summaries.append(row)
        print(
            f"320_{name} | overall_peak={pct(row['overall_peak_gt'])} | "
            f"thread_side_peak={pct(row['thread_side_peak_gt'])} | "
            f"scratch_head_peak={pct(row['scratch_head_peak_gt'])} | "
            f"meanIoU={row['overall_mean_iou']:.4f}"
        )

    if not summaries:
        raise RuntimeError("No successful layer-comparison experiments.")

    summary_path = args.work_dir / "layer_comparison_summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
        writer.writeheader()
        writer.writerows(summaries)

    print("\n=============== 特征层汇总 ===============")
    for row in summaries:
        print(
            f"320_{row['experiment']:<3} | "
            f"overall_peak={pct(row['overall_peak_gt']):>7} | "
            f"thread_side_peak={pct(row['thread_side_peak_gt']):>7} | "
            f"scratch_head_peak={pct(row['scratch_head_peak_gt']):>7} | "
            f"IoU>=0.50={pct(row['overall_iou50']):>7} | "
            f"meanIoU={row['overall_mean_iou']:.4f}"
        )

    best = max(
        summaries,
        key=lambda r: (
            r["thread_side_peak_gt"],
            r["overall_peak_gt"],
            r["scratch_head_peak_gt"],
            r["overall_mean_iou"],
        ),
    )

    print("\n=============== 当前优选 ===============")
    print(f"实验: 320_{best['experiment']}")
    print(f"layers: {best['layers']}")
    print(f"thread_side peak: {pct(best['thread_side_peak_gt'])}")
    print(f"overall peak: {pct(best['overall_peak_gt'])}")
    print(f"scratch_head peak: {pct(best['scratch_head_peak_gt'])}")
    print(f"summary CSV: {summary_path.resolve()}")
    print("=========================================")


if __name__ == "__main__":
    main()
