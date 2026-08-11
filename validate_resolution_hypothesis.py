from __future__ import annotations

import argparse
import csv
import os
import pickle
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
FIXED_LAYERS = ("layer2", "layer3")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate whether downsampling destroys small-defect information by "
            "changing only PatchCore input resolution while keeping all other "
            "settings fixed."
        )
    )
    parser.add_argument("--normal-dir", default="data/screw/train/good")
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--ground-truth-dir", default="data/screw/ground_truth")
    parser.add_argument(
        "--resolutions",
        nargs="+",
        type=int,
        default=[224, 320, 448],
        help="Square PatchCore crop sizes to compare. Example: 224 320 448 640",
    )
    parser.add_argument(
        "--baseline-model-dir",
        default="products/screw/models/patchcore",
        help="Existing 224 baseline model directory.",
    )
    parser.add_argument(
        "--work-dir",
        default="outputs/screw/resolution_validation",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain non-baseline resolution models even when matching models exist.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    print("\n$ " + " ".join(command))
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    completed = subprocess.run(command, cwd=ROOT, env=env)
    return int(completed.returncode)


def resize_for_imagesize(imagesize: int) -> int:
    return max(imagesize, int(round(imagesize * (256.0 / 224.0))))


def model_dir_for(resolution: int, baseline_model_dir: Path) -> Path:
    if resolution == 224:
        return baseline_model_dir
    return Path(f"products/screw/models/patchcore_{resolution}_l23")


def model_matches(model_dir: Path, resolution: int) -> bool:
    params_file = model_dir / "patchcore_params.pkl"
    index_file = model_dir / "nnscorer_search_index.faiss"
    if not params_file.exists() or not index_file.exists():
        return False

    try:
        with open(params_file, "rb") as f:
            saved = pickle.load(f)
        input_shape = saved.get("input_shape")
        layers = tuple(saved.get("layers_to_extract_from", ()))
        if input_shape is None:
            return False
        return int(input_shape[-1]) == resolution and layers == FIXED_LAYERS
    except Exception:
        return False


def train_resolution_model(model_dir: Path, resolution: int, args) -> bool:
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
        str(resolution),
        "--layers",
        *FIXED_LAYERS,
    ]
    if args.device:
        command.extend(["--device", args.device])
    return run_command(command) == 0


def evaluate_resolution_model(model_dir: Path, resolution: int, args) -> Path | None:
    output_dir = args.work_dir / f"eval_{resolution}"
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


def collect_defect_images(test_dir: Path) -> list[Path]:
    images = []
    for path in test_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = path.relative_to(test_dir)
        if len(rel.parts) < 2 or rel.parts[0].lower() == "good":
            continue
        images.append(path)
    return sorted(images, key=lambda p: p.as_posix().lower())


def mask_path_for(image_path: Path, test_dir: Path, gt_dir: Path) -> Path:
    rel = image_path.relative_to(test_dir)
    return gt_dir / rel.parts[0] / f"{image_path.stem}_mask.png"


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def geometry_for_mask(mask_path: Path, resolution: int) -> dict:
    resize = resize_for_imagesize(resolution)
    source = Image.open(mask_path).convert("L")

    resize_transform = transforms.Resize(
        resize,
        interpolation=InterpolationMode.NEAREST,
    )
    crop_transform = transforms.CenterCrop(resolution)

    resized = resize_transform(source)
    resized_mask = np.asarray(resized, dtype=np.uint8) > 0
    cropped = crop_transform(resized)
    crop_mask = np.asarray(cropped, dtype=np.uint8) > 0

    resized_area = int(resized_mask.sum())
    crop_area = int(crop_mask.sum())
    retention = crop_area / resized_area if resized_area > 0 else 0.0

    bbox = bbox_from_mask(crop_mask)
    if bbox is None:
        width = 0
        height = 0
    else:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

    short_side = min(width, height) if width > 0 and height > 0 else 0
    long_side = max(width, height) if width > 0 and height > 0 else 0

    return {
        "resize": resize,
        "gt_area_px": crop_area,
        "gt_bbox_width_px": width,
        "gt_bbox_height_px": height,
        "gt_bbox_short_side_px": short_side,
        "gt_bbox_long_side_px": long_side,
        "crop_retention_ratio": retention,
        # WideResNet layer2/layer3 are approximately stride 8/16 relative to input.
        "approx_layer2_short_side_cells": short_side / 8.0,
        "approx_layer3_short_side_cells": short_side / 16.0,
    }


def load_eval_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status", "ok") != "ok":
                continue
            try:
                row["_iou"] = float(row["bbox_iou"])
                row["_peak"] = str(row.get("peak_in_gt_bbox", "")).strip() in {
                    "1",
                    "true",
                    "True",
                }
            except Exception:
                continue
            rows.append(row)
    return rows


def median(values):
    return float(statistics.median(values)) if values else 0.0


def summarize_resolution(resolution: int, eval_rows: list[dict], geometry: dict[str, dict]) -> dict:
    merged = []
    for row in eval_rows:
        rel = row["relative_path"]
        geom = geometry.get(rel)
        if geom is None:
            continue
        merged.append((row, geom))

    if not merged:
        raise RuntimeError(f"No matched rows for resolution {resolution}")

    def metrics(items):
        if not items:
            return {
                "count": 0,
                "mean_iou": 0.0,
                "iou50": 0.0,
                "peak_gt": 0.0,
                "median_gt_area_px": 0.0,
                "median_gt_short_side_px": 0.0,
                "median_l2_cells": 0.0,
                "median_l3_cells": 0.0,
                "mean_crop_retention": 0.0,
            }
        return {
            "count": len(items),
            "mean_iou": sum(r["_iou"] for r, _ in items) / len(items),
            "iou50": sum(r["_iou"] >= 0.50 for r, _ in items) / len(items),
            "peak_gt": sum(r["_peak"] for r, _ in items) / len(items),
            "median_gt_area_px": median([g["gt_area_px"] for _, g in items]),
            "median_gt_short_side_px": median([g["gt_bbox_short_side_px"] for _, g in items]),
            "median_l2_cells": median([g["approx_layer2_short_side_cells"] for _, g in items]),
            "median_l3_cells": median([g["approx_layer3_short_side_cells"] for _, g in items]),
            "mean_crop_retention": sum(g["crop_retention_ratio"] for _, g in items) / len(items),
        }

    summary = {
        "resolution": resolution,
        "resize": resize_for_imagesize(resolution),
        "overall": metrics(merged),
    }

    for defect_type in [
        "manipulated_front",
        "scratch_head",
        "scratch_neck",
        "thread_side",
        "thread_top",
    ]:
        subset = [item for item in merged if item[0].get("defect_type") == defect_type]
        summary[defect_type] = metrics(subset)

    areas = np.array([g["gt_area_px"] for _, g in merged], dtype=np.float32)
    q25 = float(np.quantile(areas, 0.25))
    q75 = float(np.quantile(areas, 0.75))
    small = [item for item in merged if item[1]["gt_area_px"] <= q25]
    large = [item for item in merged if item[1]["gt_area_px"] >= q75]
    summary["small_quartile"] = metrics(small)
    summary["large_quartile"] = metrics(large)

    return summary


def flatten_summary(summary: dict) -> dict:
    overall = summary["overall"]
    thread_side = summary["thread_side"]
    scratch_head = summary["scratch_head"]
    small = summary["small_quartile"]
    large = summary["large_quartile"]

    return {
        "resolution": summary["resolution"],
        "resize": summary["resize"],
        "overall_peak_gt": overall["peak_gt"],
        "overall_mean_iou": overall["mean_iou"],
        "overall_iou50": overall["iou50"],
        "thread_side_peak_gt": thread_side["peak_gt"],
        "thread_side_mean_iou": thread_side["mean_iou"],
        "scratch_head_peak_gt": scratch_head["peak_gt"],
        "scratch_head_mean_iou": scratch_head["mean_iou"],
        "small_quartile_peak_gt": small["peak_gt"],
        "large_quartile_peak_gt": large["peak_gt"],
        "median_gt_area_px": overall["median_gt_area_px"],
        "median_gt_short_side_px": overall["median_gt_short_side_px"],
        "median_layer2_short_side_cells": overall["median_l2_cells"],
        "median_layer3_short_side_cells": overall["median_l3_cells"],
        "mean_crop_retention": overall["mean_crop_retention"],
    }


def write_sample_diagnostics(
    output_path: Path,
    resolution: int,
    eval_rows: list[dict],
    geometry: dict[str, dict],
):
    rows = []
    for row in eval_rows:
        rel = row["relative_path"]
        geom = geometry.get(rel)
        if geom is None:
            continue
        rows.append(
            {
                "resolution": resolution,
                "relative_path": rel,
                "defect_type": row.get("defect_type", ""),
                "anomaly_score": row.get("anomaly_score", ""),
                "bbox_iou": row.get("bbox_iou", ""),
                "peak_in_gt_bbox": row.get("peak_in_gt_bbox", ""),
                **geom,
            }
        )

    if not rows:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def print_summary(row: dict):
    print(
        f"{int(row['resolution']):>4}px | "
        f"overall_peak={row['overall_peak_gt']*100:6.2f}% | "
        f"thread_side_peak={row['thread_side_peak_gt']*100:6.2f}% | "
        f"scratch_head_peak={row['scratch_head_peak_gt']*100:6.2f}% | "
        f"small25_peak={row['small_quartile_peak_gt']*100:6.2f}% | "
        f"meanIoU={row['overall_mean_iou']:.4f}"
    )


def interpret(rows: list[dict]):
    ordered = sorted(rows, key=lambda r: int(r["resolution"]))
    baseline = ordered[0]
    highest = ordered[-1]

    overall_gain = highest["overall_peak_gt"] - baseline["overall_peak_gt"]
    thread_gain = highest["thread_side_peak_gt"] - baseline["thread_side_peak_gt"]
    scratch_gain = highest["scratch_head_peak_gt"] - baseline["scratch_head_peak_gt"]
    small_gain = highest["small_quartile_peak_gt"] - baseline["small_quartile_peak_gt"]
    baseline_size_gap = baseline["large_quartile_peak_gt"] - baseline["small_quartile_peak_gt"]

    print("\n=============== 像素/分辨率假设验证 ===============")
    print(
        f"{int(baseline['resolution'])} -> {int(highest['resolution'])} overall peak命中变化: "
        f"{overall_gain*100:+.2f} 个百分点"
    )
    print(f"thread_side peak命中变化: {thread_gain*100:+.2f} 个百分点")
    print(f"scratch_head peak命中变化: {scratch_gain*100:+.2f} 个百分点")
    print(f"最小25%缺陷 peak命中变化: {small_gain*100:+.2f} 个百分点")
    print(
        f"224基线 大缺陷25% vs 小缺陷25% peak命中差: "
        f"{baseline_size_gap*100:+.2f} 个百分点"
    )
    print(
        f"224基线 median GT短边: {baseline['median_gt_short_side_px']:.1f}px, "
        f"约占layer2 {baseline['median_layer2_short_side_cells']:.2f} cells, "
        f"layer3 {baseline['median_layer3_short_side_cells']:.2f} cells"
    )

    support_count = 0
    if overall_gain >= 0.03:
        support_count += 1
    if thread_gain >= 0.08 or scratch_gain >= 0.08:
        support_count += 1
    if small_gain >= 0.08:
        support_count += 1
    if baseline_size_gap >= 0.10:
        support_count += 1

    if support_count >= 3:
        verdict = "强支持：分辨率降低/下采样很可能是特征不完整和定位错位的重要原因。"
    elif support_count >= 1:
        verdict = "部分支持：分辨率是影响因素之一，但还可能同时存在纹理层级、裁剪或背景响应问题。"
    else:
        verdict = "当前证据不足：仅提高输入分辨率没有带来稳定收益，需要转向特征层/ROI/切块方案验证。"

    print("结论:", verdict)

    low_retention = baseline["mean_crop_retention"] < 0.95
    if low_retention:
        print(
            "另外：平均GT裁剪保留率低于95%，CenterCrop 也可能在丢失缺陷信息，"
            "需要与纯Resize问题分开处理。"
        )
    print("===================================================")


def main():
    args = parse_args()
    args.normal_dir = Path(args.normal_dir)
    args.test_dir = Path(args.test_dir)
    args.ground_truth_dir = Path(args.ground_truth_dir)
    args.baseline_model_dir = Path(args.baseline_model_dir)
    args.work_dir = Path(args.work_dir)

    resolutions = sorted(set(args.resolutions))
    if not resolutions or resolutions[0] != 224:
        raise ValueError("Validation must include 224 as the baseline resolution.")

    args.work_dir.mkdir(parents=True, exist_ok=True)
    sample_csv = args.work_dir / "sample_resolution_diagnostics.csv"
    if sample_csv.exists():
        sample_csv.unlink()

    defect_images = collect_defect_images(args.test_dir)
    if not defect_images:
        raise RuntimeError(f"No defect images found under {args.test_dir}")

    geometry_by_resolution = {}
    for resolution in resolutions:
        geometry = {}
        for image_path in defect_images:
            gt_path = mask_path_for(image_path, args.test_dir, args.ground_truth_dir)
            if not gt_path.exists():
                continue
            rel = image_path.relative_to(args.test_dir).as_posix()
            geometry[rel] = geometry_for_mask(gt_path, resolution)
        geometry_by_resolution[resolution] = geometry

    summary_rows = []

    print("========== 分辨率单因素验证 ==========")
    print(f"resolutions: {resolutions}")
    print(f"固定 layers: {FIXED_LAYERS}")
    print(f"固定 coreset: {args.coreset}")
    print("除输入分辨率外，其余PatchCore配置保持一致。")
    print("====================================")

    for resolution in resolutions:
        model_dir = model_dir_for(resolution, args.baseline_model_dir)

        if resolution == 224:
            if not model_matches(model_dir, resolution):
                raise RuntimeError(
                    f"224 baseline model is missing or does not match layer2/layer3: {model_dir}"
                )
        else:
            need_train = args.force_retrain or not model_matches(model_dir, resolution)
            if need_train:
                if model_dir.exists():
                    shutil.rmtree(model_dir)
                print(f"\n训练 {resolution}px 单因素模型: {model_dir}")
                if not train_resolution_model(model_dir, resolution, args):
                    print(f"{resolution}px 训练失败，跳过。")
                    continue
            else:
                print(f"\n复用已有匹配模型: {model_dir}")

        eval_csv = evaluate_resolution_model(model_dir, resolution, args)
        if eval_csv is None:
            print(f"{resolution}px 评估失败，跳过。")
            continue

        eval_rows = load_eval_rows(eval_csv)
        summary = summarize_resolution(
            resolution,
            eval_rows,
            geometry_by_resolution[resolution],
        )
        row = flatten_summary(summary)
        summary_rows.append(row)
        write_sample_diagnostics(
            sample_csv,
            resolution,
            eval_rows,
            geometry_by_resolution[resolution],
        )
        print_summary(row)

    if len(summary_rows) < 2:
        raise RuntimeError("Need at least two completed resolutions to validate the hypothesis.")

    summary_csv = args.work_dir / "resolution_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n=============== 分辨率汇总 ===============")
    for row in sorted(summary_rows, key=lambda r: int(r["resolution"])):
        print_summary(row)
    print(f"汇总CSV: {summary_csv.resolve()}")
    print(f"逐样本诊断: {sample_csv.resolve()}")

    interpret(summary_rows)


if __name__ == "__main__":
    main()
