from __future__ import annotations

import argparse
import csv
import os
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
RESOLUTION = 320
LAYERS = ("layer2", "layer3")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare 320px PatchCore input strategies while keeping backbone layers "
            "and coreset fixed: baseline center-crop, direct resize, and automatic product ROI."
        )
    )
    parser.add_argument("--normal-dir", default="data/screw/train/good")
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--ground-truth-dir", default="data/screw/ground_truth")
    parser.add_argument("--baseline-model-dir", default="products/screw/models/patchcore_320_l23")
    parser.add_argument("--direct-model-dir", default="products/screw/models/patchcore_320_direct_l23")
    parser.add_argument("--roi-model-dir", default="products/screw/models/patchcore_320_roi_l23")
    parser.add_argument("--work-dir", default="outputs/screw/roi_preprocessing_320")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--bbox-relative-threshold", type=float, default=0.80)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-rebuild-roi", action="store_true")
    parser.add_argument(
        "--roi-margin",
        type=float,
        default=0.10,
        help="Relative margin added around automatically detected product bbox.",
    )
    return parser.parse_args()


def run_command(command: list[str]) -> int:
    print("\n$ " + " ".join(command))
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    result = subprocess.run(command, cwd=ROOT, env=env)
    return int(result.returncode)


def model_complete(model_dir: Path) -> bool:
    return (
        (model_dir / "patchcore_params.pkl").exists()
        and (model_dir / "nnscorer_search_index.faiss").exists()
    )


def train_model(normal_dir: Path, model_dir: Path, resize: int, args) -> bool:
    command = [
        sys.executable,
        "train_patchcore.py",
        "--normal-dir",
        str(normal_dir),
        "--model-dir",
        str(model_dir),
        "--batch-size",
        str(args.batch_size),
        "--coreset",
        str(args.coreset),
        "--imagesize",
        str(RESOLUTION),
        "--resize",
        str(resize),
        "--layers",
        *LAYERS,
    ]
    if args.device:
        command.extend(["--device", args.device])
    return run_command(command) == 0


def evaluate_model(test_dir: Path, gt_dir: Path, model_dir: Path, output_dir: Path, args) -> Path | None:
    command = [
        sys.executable,
        "evaluate_mvtec_screw.py",
        "--test-dir",
        str(test_dir),
        "--ground-truth-dir",
        str(gt_dir),
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


def iter_images(folder: Path):
    for path in sorted(folder.rglob("*"), key=lambda p: p.as_posix().lower()):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


def border_background_rgb(rgb: np.ndarray) -> np.ndarray:
    h, w = rgb.shape[:2]
    bw = max(2, min(h, w) // 40)
    border = np.concatenate(
        [
            rgb[:bw].reshape(-1, 3),
            rgb[-bw:].reshape(-1, 3),
            rgb[:, :bw].reshape(-1, 3),
            rgb[:, -bw:].reshape(-1, 3),
        ],
        axis=0,
    )
    return np.median(border.astype(np.float32), axis=0)


def detect_product_bbox(image: Image.Image, margin: float = 0.10):
    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    h, w = rgb.shape[:2]
    if h < 8 or w < 8:
        return (0, 0, w, h), False, 1.0

    bg = border_background_rgb(rgb)
    dist = np.linalg.norm(rgb.astype(np.float32) - bg[None, None, :], axis=2)

    p99 = float(np.percentile(dist, 99.0))
    if p99 <= 1e-6:
        return (0, 0, w, h), False, 1.0

    dist_u8 = np.clip(dist * (255.0 / p99), 0, 255).astype(np.uint8)
    otsu, binary = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Avoid a too-permissive Otsu threshold on nearly uniform backgrounds.
    min_thr = max(6, int(round(0.07 * 255.0)))
    if otsu < min_thr:
        binary = (dist_u8 >= min_thr).astype(np.uint8) * 255

    kernel_close = np.ones((7, 7), np.uint8)
    kernel_open = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return (0, 0, w, h), False, 1.0

    candidates = []
    image_area = float(h * w)
    for label_id in range(1, n):
        x, y, bw, bh, area = stats[label_id]
        area_ratio = float(area) / image_area
        if area_ratio < 0.002:
            continue
        candidates.append((int(area), int(x), int(y), int(bw), int(bh)))

    if not candidates:
        return (0, 0, w, h), False, 1.0

    _, x, y, bw, bh = max(candidates, key=lambda item: item[0])
    x1, y1, x2, y2 = x, y, x + bw, y + bh

    pad_x = int(round(bw * max(0.0, margin)))
    pad_y = int(round(bh * max(0.0, margin)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)

    roi_area_ratio = ((x2 - x1) * (y2 - y1)) / image_area

    # Reject obviously bad segmentation: near-full-frame or implausibly tiny ROI.
    if roi_area_ratio > 0.95 or roi_area_ratio < 0.01:
        return (0, 0, w, h), False, 1.0

    return (x1, y1, x2, y2), True, float(roi_area_ratio)


def crop_square(image: Image.Image, bbox, fill):
    x1, y1, x2, y2 = bbox
    crop = image.crop((x1, y1, x2, y2))
    width, height = crop.size
    side = max(width, height)
    if side <= 0:
        return image.copy()

    left = (side - width) // 2
    top = (side - height) // 2
    if image.mode == "L":
        canvas = Image.new("L", (side, side), color=int(fill))
    else:
        canvas = Image.new("RGB", (side, side), color=tuple(int(v) for v in fill))
    canvas.paste(crop, (left, top))
    return canvas


def save_roi_pair(image_path: Path, output_path: Path, margin: float, mask_path: Path | None = None, mask_output_path: Path | None = None):
    image = Image.open(image_path).convert("RGB")
    rgb = np.asarray(image, dtype=np.uint8)
    bg = border_background_rgb(rgb)
    bbox, detected, roi_area_ratio = detect_product_bbox(image, margin=margin)

    roi = crop_square(image, bbox, fill=bg)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    roi.save(output_path)

    if mask_path is not None and mask_output_path is not None:
        mask = Image.open(mask_path).convert("L")
        mask_roi = crop_square(mask, bbox, fill=0)
        mask_output_path.parent.mkdir(parents=True, exist_ok=True)
        mask_roi.save(mask_output_path)

    return bbox, detected, roi_area_ratio, image.size, roi.size


def build_roi_dataset(args):
    work_dir = Path(args.work_dir)
    roi_root = work_dir / "roi_data"
    train_out = roi_root / "train" / "good"
    test_out = roi_root / "test"
    gt_out = roi_root / "ground_truth"
    metadata_csv = work_dir / "roi_metadata.csv"

    if args.force_rebuild_roi and roi_root.exists():
        shutil.rmtree(roi_root)

    if roi_root.exists() and metadata_csv.exists() and train_out.exists() and test_out.exists() and gt_out.exists():
        print(f"已有ROI数据，跳过重建: {roi_root}")
        return train_out, test_out, gt_out, metadata_csv

    if roi_root.exists():
        shutil.rmtree(roi_root)

    rows = []

    normal_dir = Path(args.normal_dir)
    for image_path in iter_images(normal_dir):
        rel = image_path.relative_to(normal_dir)
        out_path = train_out / rel
        bbox, detected, ratio, original_size, roi_size = save_roi_pair(
            image_path, out_path, margin=args.roi_margin
        )
        rows.append(
            {
                "split": "train_good",
                "relative_path": rel.as_posix(),
                "detected": int(detected),
                "bbox": ",".join(str(v) for v in bbox),
                "roi_area_ratio": f"{ratio:.6f}",
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "roi_size": f"{roi_size[0]}x{roi_size[1]}",
            }
        )

    test_dir = Path(args.test_dir)
    gt_dir = Path(args.ground_truth_dir)
    for image_path in iter_images(test_dir):
        rel = image_path.relative_to(test_dir)
        out_path = test_out / rel

        mask_path = None
        mask_out = None
        if len(rel.parts) >= 2 and rel.parts[0].lower() != "good":
            mask_path = gt_dir / rel.parts[0] / f"{image_path.stem}_mask.png"
            mask_out = gt_out / rel.parts[0] / f"{image_path.stem}_mask.png"
            if not mask_path.exists():
                mask_path = None
                mask_out = None

        bbox, detected, ratio, original_size, roi_size = save_roi_pair(
            image_path,
            out_path,
            margin=args.roi_margin,
            mask_path=mask_path,
            mask_output_path=mask_out,
        )
        rows.append(
            {
                "split": "test",
                "relative_path": rel.as_posix(),
                "detected": int(detected),
                "bbox": ",".join(str(v) for v in bbox),
                "roi_area_ratio": f"{ratio:.6f}",
                "original_size": f"{original_size[0]}x{original_size[1]}",
                "roi_size": f"{roi_size[0]}x{roi_size[1]}",
            }
        )

    work_dir.mkdir(parents=True, exist_ok=True)
    with open(metadata_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    detected_count = sum(int(r["detected"]) for r in rows)
    ratios = [float(r["roi_area_ratio"]) for r in rows if int(r["detected"]) == 1]
    print("\n========== ROI数据构建 ==========")
    print(f"总图片: {len(rows)}")
    print(f"成功检测产品ROI: {detected_count}/{len(rows)} ({detected_count / max(1, len(rows)):.2%})")
    if ratios:
        print(f"ROI面积占原图中位数: {statistics.median(ratios):.2%}")
        print(f"等效线性放大约: {1.0 / max(1e-6, statistics.median(ratios) ** 0.5):.2f}x")
    print(f"ROI metadata: {metadata_csv.resolve()}")
    print("================================")

    return train_out, test_out, gt_out, metadata_csv


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_rows(csv_path: Path):
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


def metrics(rows):
    if not rows:
        return {"peak": 0.0, "mean_iou": 0.0, "iou50": 0.0}
    return {
        "peak": sum(r["_peak"] for r in rows) / len(rows),
        "mean_iou": sum(r["_iou"] for r in rows) / len(rows),
        "iou50": sum(r["_iou"] >= 0.50 for r in rows) / len(rows),
    }


def summarize(name: str, csv_path: Path):
    rows = load_rows(csv_path)
    overall = metrics(rows)
    thread_side = metrics([r for r in rows if r.get("defect_type") == "thread_side"])
    scratch_head = metrics([r for r in rows if r.get("defect_type") == "scratch_head"])
    manipulated_front = metrics([r for r in rows if r.get("defect_type") == "manipulated_front"])
    return {
        "experiment": name,
        "overall_peak_gt": overall["peak"],
        "overall_mean_iou": overall["mean_iou"],
        "overall_iou50": overall["iou50"],
        "thread_side_peak_gt": thread_side["peak"],
        "thread_side_mean_iou": thread_side["mean_iou"],
        "scratch_head_peak_gt": scratch_head["peak"],
        "scratch_head_mean_iou": scratch_head["mean_iou"],
        "manipulated_front_peak_gt": manipulated_front["peak"],
    }


def pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def main():
    args = parse_args()
    args.work_dir = Path(args.work_dir)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    normal_dir = Path(args.normal_dir)
    test_dir = Path(args.test_dir)
    gt_dir = Path(args.ground_truth_dir)
    baseline_model = Path(args.baseline_model_dir)
    direct_model = Path(args.direct_model_dir)
    roi_model = Path(args.roi_model_dir)

    print("========== 320px 输入策略单因素比较 ==========")
    print("固定 imagesize: 320")
    print("固定 layers: layer2 + layer3")
    print(f"固定 coreset: {args.coreset}")
    print("A: baseline = resize366 + center crop320")
    print("B: direct   = direct resize320, no extra crop")
    print("C: ROI      = image-only product ROI + square pad + direct resize320")
    print("GT仅用于最终评估，ROI检测不读取GT。")
    print("============================================")

    roi_train, roi_test, roi_gt, _ = build_roi_dataset(args)

    experiments = []

    # A: current best baseline. Rebuild only if missing or force requested.
    if args.force_retrain or not model_complete(baseline_model):
        print("\n训练 baseline 320_l23 ...")
        if not train_model(normal_dir, baseline_model, resize=366, args=args):
            raise RuntimeError("Baseline model training failed")
    baseline_csv = evaluate_model(
        test_dir,
        gt_dir,
        baseline_model,
        args.work_dir / "eval_baseline_crop320",
        args,
    )
    if baseline_csv:
        experiments.append(summarize("baseline_crop320", baseline_csv))

    # B: same full image, but no 366->center-crop. This isolates crop/scale behavior.
    if args.force_retrain or not model_complete(direct_model):
        print("\n训练 direct-resize 320_l23 ...")
        if not train_model(normal_dir, direct_model, resize=320, args=args):
            raise RuntimeError("Direct-resize model training failed")
    direct_csv = evaluate_model(
        test_dir,
        gt_dir,
        direct_model,
        args.work_dir / "eval_direct320",
        args,
    )
    if direct_csv:
        experiments.append(summarize("direct320", direct_csv))

    # C: product ROI derived only from the image, then direct resize to 320.
    if args.force_retrain or not model_complete(roi_model):
        print("\n训练 ROI 320_l23 ...")
        if not train_model(roi_train, roi_model, resize=320, args=args):
            raise RuntimeError("ROI model training failed")
    roi_csv = evaluate_model(
        roi_test,
        roi_gt,
        roi_model,
        args.work_dir / "eval_roi320",
        args,
    )
    if roi_csv:
        experiments.append(summarize("roi320", roi_csv))

    if not experiments:
        raise RuntimeError("No successful preprocessing experiments")

    summary_csv = args.work_dir / "preprocessing_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(experiments[0].keys()))
        writer.writeheader()
        writer.writerows(experiments)

    print("\n=============== 输入策略汇总 ===============")
    for row in experiments:
        print(
            f"{row['experiment']:<18} | "
            f"overall_peak={pct(row['overall_peak_gt']):>7} | "
            f"thread_side_peak={pct(row['thread_side_peak_gt']):>7} | "
            f"scratch_head_peak={pct(row['scratch_head_peak_gt']):>7} | "
            f"IoU>=0.50={pct(row['overall_iou50']):>7} | "
            f"meanIoU={row['overall_mean_iou']:.4f}"
        )

    best = max(
        experiments,
        key=lambda r: (
            r["thread_side_peak_gt"],
            r["overall_peak_gt"],
            r["scratch_head_peak_gt"],
            r["overall_mean_iou"],
        ),
    )

    print("\n=============== 当前优选 ===============")
    print(f"实验: {best['experiment']}")
    print(f"thread_side peak: {pct(best['thread_side_peak_gt'])}")
    print(f"overall peak: {pct(best['overall_peak_gt'])}")
    print(f"scratch_head peak: {pct(best['scratch_head_peak_gt'])}")
    print(f"summary CSV: {summary_csv.resolve()}")
    print("=========================================")


if __name__ == "__main__":
    main()
