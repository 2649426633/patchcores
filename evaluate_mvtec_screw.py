from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from app.anomaly.patchcore_adapter import PatchCoreAdapter
from app.anomaly.postprocessing import extract_bbox_from_map, normalize_anomaly_map
from app.anomaly.preprocessing import load_display_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PatchCore localization on MVTec screw using ground-truth masks. "
            "Ground truth is used for evaluation only, never for training/inference."
        )
    )
    parser.add_argument("--test-dir", default="data/screw/test")
    parser.add_argument("--ground-truth-dir", default="data/screw/ground_truth")
    parser.add_argument("--model-dir", default="products/screw/models/patchcore")
    parser.add_argument("--output-dir", default="outputs/screw/localization_eval")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument(
        "--bbox-relative-threshold",
        type=float,
        default=0.80,
        help="Localization-only threshold used by the current bbox postprocessing.",
    )
    parser.add_argument(
        "--top-percent",
        type=float,
        default=1.0,
        help="Percentage of hottest anomaly-map pixels used for top-region overlap diagnostics.",
    )
    return parser.parse_args()


def collect_defect_images(test_dir: Path) -> list[Path]:
    images = []
    for p in test_dir.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        rel = p.relative_to(test_dir)
        if len(rel.parts) < 2 or rel.parts[0].lower() == "good":
            continue
        images.append(p)
    return sorted(images, key=lambda p: p.as_posix().lower())


def mask_path_for(image_path: Path, test_dir: Path, ground_truth_dir: Path) -> Path:
    rel = image_path.relative_to(test_dir)
    defect_type = rel.parts[0]
    stem = image_path.stem
    return ground_truth_dir / defect_type / f"{stem}_mask.png"


def preprocess_mask(mask_path: Path, resize: int, imagesize: int) -> np.ndarray:
    mask = Image.open(mask_path).convert("L")
    transform = transforms.Compose(
        [
            transforms.Resize(resize, interpolation=InterpolationMode.NEAREST),
            transforms.CenterCrop(imagesize),
        ]
    )
    mask = transform(mask)
    return np.asarray(mask, dtype=np.uint8) > 0


def bbox_from_binary(mask: np.ndarray):
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def bbox_iou(a, b) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def point_in_bbox(x: int, y: int, bbox) -> bool:
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return x1 <= x < x2 and y1 <= y < y2


def save_diagnostic_overlay(
    image: Image.Image,
    pred_bbox,
    gt_bbox,
    peak_xy,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.asarray(image.convert("RGB"))
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    if gt_bbox is not None:
        x1, y1, x2, y2 = gt_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)

    if pred_bbox is not None:
        x1, y1, x2, y2 = pred_bbox
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 255, 255), 2)

    px, py = peak_xy
    cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1)

    cv2.imwrite(str(output_path), canvas)


def main():
    args = parse_args()

    test_dir = Path(args.test_dir)
    gt_dir = Path(args.ground_truth_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir.resolve()}")

    images = collect_defect_images(test_dir)
    if not images:
        raise RuntimeError(f"No defect test images found under: {test_dir.resolve()}")

    detector = PatchCoreAdapter(device=args.device)
    detector.load(model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "localization_results.csv"

    rows = []
    top_fraction = max(1e-6, min(1.0, args.top_percent / 100.0))

    print("========== MVTec Screw 定位客观评估 ==========")
    print(f"缺陷图片数: {len(images)}")
    print(f"BBox相对阈值: {args.bbox_relative_threshold:.2f}")
    print(f"Top热区比例: {args.top_percent:.2f}%")
    print("绿色框=GT，白色框=预测，红点=anomaly map峰值")
    print("============================================")

    for i, image_path in enumerate(images, start=1):
        rel = image_path.relative_to(test_dir)
        defect_type = rel.parts[0]
        gt_path = mask_path_for(image_path, test_dir, gt_dir)

        row = {
            "index": i,
            "relative_path": rel.as_posix(),
            "defect_type": defect_type,
            "anomaly_score": "",
            "pred_bbox": "",
            "gt_bbox": "",
            "bbox_iou": "",
            "peak_x": "",
            "peak_y": "",
            "peak_in_gt_mask": "",
            "peak_in_gt_bbox": "",
            "top_region_precision": "",
            "top_region_recall": "",
            "diagnostic_overlay": "",
            "status": "ok",
            "error": "",
        }

        try:
            if not gt_path.exists():
                raise FileNotFoundError(f"GT mask missing: {gt_path}")

            result = detector.predict(image_path)
            anomaly_map = np.asarray(result["anomaly_map"], dtype=np.float32)
            norm = normalize_anomaly_map(anomaly_map)

            gt_mask = preprocess_mask(
                gt_path,
                resize=detector.config.resize,
                imagesize=detector.config.imagesize,
            )
            if gt_mask.shape != norm.shape:
                gt_mask = cv2.resize(
                    gt_mask.astype(np.uint8),
                    (norm.shape[1], norm.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            pred_bbox = extract_bbox_from_map(
                anomaly_map,
                relative_threshold=args.bbox_relative_threshold,
            )
            gt_bbox = bbox_from_binary(gt_mask)
            iou = bbox_iou(pred_bbox, gt_bbox)

            peak_index = int(np.argmax(norm))
            peak_y, peak_x = np.unravel_index(peak_index, norm.shape)
            peak_in_mask = bool(gt_mask[peak_y, peak_x])
            peak_in_bbox = point_in_bbox(int(peak_x), int(peak_y), gt_bbox)

            flat = norm.reshape(-1)
            k = max(1, int(round(flat.size * top_fraction)))
            top_indices = np.argpartition(flat, -k)[-k:]
            top_mask = np.zeros_like(flat, dtype=bool)
            top_mask[top_indices] = True
            top_mask = top_mask.reshape(norm.shape)

            intersection = int(np.logical_and(top_mask, gt_mask).sum())
            top_precision = intersection / max(1, int(top_mask.sum()))
            top_recall = intersection / max(1, int(gt_mask.sum()))

            display_image = load_display_image(
                image_path,
                resize=detector.config.resize,
                imagesize=detector.config.imagesize,
            )
            diagnostic_path = output_dir / defect_type / f"{image_path.stem}_diagnostic.jpg"
            save_diagnostic_overlay(
                display_image,
                pred_bbox,
                gt_bbox,
                (int(peak_x), int(peak_y)),
                diagnostic_path,
            )

            def bbox_text(b):
                return "" if b is None else ",".join(str(v) for v in b)

            row.update(
                {
                    "anomaly_score": f"{result['anomaly_score']:.6f}",
                    "pred_bbox": bbox_text(pred_bbox),
                    "gt_bbox": bbox_text(gt_bbox),
                    "bbox_iou": f"{iou:.6f}",
                    "peak_x": int(peak_x),
                    "peak_y": int(peak_y),
                    "peak_in_gt_mask": int(peak_in_mask),
                    "peak_in_gt_bbox": int(peak_in_bbox),
                    "top_region_precision": f"{top_precision:.6f}",
                    "top_region_recall": f"{top_recall:.6f}",
                    "diagnostic_overlay": str(diagnostic_path.resolve()),
                }
            )

            verdict = "MAP错" if not peak_in_bbox else ("BBox偏" if iou < 0.20 else "OK")
            print(
                f"[{i:03d}/{len(images):03d}] {rel.as_posix()}  "
                f"score={result['anomaly_score']:.3f}  IoU={iou:.3f}  "
                f"peak_in_GT={peak_in_bbox}  => {verdict}"
            )

        except Exception as exc:
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{i:03d}/{len(images):03d}] {rel.as_posix()} ERROR: {row['error']}")

        rows.append(row)

        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    ok_rows = [r for r in rows if r["status"] == "ok"]
    if ok_rows:
        ious = np.array([float(r["bbox_iou"]) for r in ok_rows], dtype=np.float32)
        peak_hits = np.array([int(r["peak_in_gt_bbox"]) for r in ok_rows], dtype=np.int32)
        print("\n=============== 汇总 ===============")
        print(f"有效样本: {len(ok_rows)}")
        print(f"平均BBox IoU: {ious.mean():.4f}")
        print(f"中位BBox IoU: {np.median(ious):.4f}")
        print(f"IoU >= 0.20: {(ious >= 0.20).mean():.2%}")
        print(f"IoU >= 0.50: {(ious >= 0.50).mean():.2%}")
        print(f"Anomaly峰值落在GT框内: {peak_hits.mean():.2%}")
        print(f"结果CSV: {csv_path.resolve()}")
        print("===================================")

        print("\n最差20张（按BBox IoU排序）:")
        worst = sorted(ok_rows, key=lambda r: float(r["bbox_iou"]))[:20]
        for r in worst:
            print(
                f"  {r['relative_path']:<35} IoU={float(r['bbox_iou']):.3f}  "
                f"peak_in_GT={r['peak_in_gt_bbox']}  score={r['anomaly_score']}"
            )


if __name__ == "__main__":
    main()
