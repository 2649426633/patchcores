from __future__ import annotations

import argparse
import csv
from pathlib import Path

from app.anomaly.patchcore_adapter import PatchCoreAdapter
from app.anomaly.postprocessing import save_heatmap, save_overlay_with_bbox
from app.anomaly.preprocessing import load_display_image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch-test every image under a PatchCore test directory."
    )
    parser.add_argument(
        "--test-dir",
        default="data/screw/test",
        help="Root test directory. All supported images below it are tested recursively.",
    )
    parser.add_argument(
        "--model-dir",
        default="products/screw/models/patchcore",
        help="Saved PatchCore model directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/screw/all_test",
        help="Directory for heatmaps, overlays and CSV results.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda or cuda:0. Default: auto-select.",
    )
    parser.add_argument(
        "--bbox-relative-threshold",
        type=float,
        default=0.80,
        help="Localization-only relative threshold. This is not a PASS/NG threshold.",
    )
    return parser.parse_args()


def collect_images(test_dir: Path) -> list[Path]:
    images = [
        p for p in test_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda p: p.as_posix().lower())


def bbox_to_text(bbox):
    if bbox is None:
        return ""
    return ",".join(str(v) for v in bbox)


def main():
    args = parse_args()

    test_dir = Path(args.test_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")

    images = collect_images(test_dir)
    if not images:
        raise RuntimeError(f"No supported test images found under: {test_dir.resolve()}")

    print("========== PatchCore 全量 Test 批量测试 ==========")
    print(f"Test目录: {test_dir.resolve()}")
    print(f"图片数量: {len(images)}")
    print(f"模型目录: {model_dir.resolve()}")
    print(f"输出目录: {output_dir.resolve()}")
    print(f"BBox相对阈值: {args.bbox_relative_threshold:.2f}")
    print("===============================================")

    # Load backbone + PatchCore + FAISS only once.
    detector = PatchCoreAdapter(device=args.device)
    detector.load(model_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "results.csv"

    rows = []
    failed = 0

    for index, image_path in enumerate(images, start=1):
        relative_path = image_path.relative_to(test_dir)
        defect_type = relative_path.parts[0] if len(relative_path.parts) > 1 else "unknown"

        # Preserve test subdirectory structure to avoid filename collisions such as
        # good/000.png and scratch_head/000.png.
        sample_output_dir = output_dir / relative_path.parent
        sample_output_dir.mkdir(parents=True, exist_ok=True)

        stem = image_path.stem
        heatmap_path = sample_output_dir / f"{stem}_heatmap.jpg"
        overlay_path = sample_output_dir / f"{stem}_overlay.jpg"

        print(f"[{index:03d}/{len(images):03d}] {relative_path.as_posix()}")

        row = {
            "index": index,
            "relative_path": relative_path.as_posix(),
            "defect_type": defect_type,
            "anomaly_score": "",
            "bbox": "",
            "heatmap_path": "",
            "overlay_path": "",
            "status": "ok",
            "error": "",
        }

        try:
            result = detector.predict(image_path)

            save_heatmap(result["anomaly_map"], heatmap_path)
            display_image = load_display_image(
                image_path,
                resize=detector.config.resize,
                imagesize=detector.config.imagesize,
            )
            _, bbox = save_overlay_with_bbox(
                display_image,
                result["anomaly_map"],
                overlay_path,
                relative_threshold=args.bbox_relative_threshold,
            )

            row["anomaly_score"] = f"{result['anomaly_score']:.6f}"
            row["bbox"] = bbox_to_text(bbox)
            row["heatmap_path"] = str(heatmap_path.resolve())
            row["overlay_path"] = str(overlay_path.resolve())

            print(
                f"    score={row['anomaly_score']}  "
                f"bbox={bbox if bbox is not None else 'None'}"
            )

        except Exception as exc:
            failed += 1
            row["status"] = "failed"
            row["error"] = f"{type(exc).__name__}: {exc}"
            print(f"    ERROR: {row['error']}")

        rows.append(row)

        # Rewrite after every sample so partial results survive an interrupted run.
        with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    success = len(images) - failed

    print("\n=============== 测试完成 ===============")
    print(f"总图片数: {len(images)}")
    print(f"成功: {success}")
    print(f"失败: {failed}")
    print(f"结果CSV: {csv_path.resolve()}")
    print(f"可视化目录: {output_dir.resolve()}")
    print("说明：当前仅输出 PatchCore anomaly score 和候选定位框，")
    print("      尚未使用最终 PASS/NG 判定阈值。")
    print("=======================================")


if __name__ == "__main__":
    main()
