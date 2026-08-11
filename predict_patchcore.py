import argparse
from pathlib import Path

from app.anomaly.patchcore_adapter import PatchCoreAdapter
from app.anomaly.postprocessing import save_heatmap, save_overlay_with_bbox
from app.anomaly.preprocessing import load_display_image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict one image with PatchCore")
    parser.add_argument("image", help="Test image path")
    parser.add_argument(
        "--model-dir",
        default="products/screw/models/patchcore",
        help="Saved PatchCore model directory",
    )
    parser.add_argument("--output-dir", default="outputs/screw")
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument(
        "--bbox-relative-threshold",
        type=float,
        default=0.70,
        help="Visualization-only relative threshold; not PASS/NG threshold",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path.resolve()}")

    detector = PatchCoreAdapter(device=args.device)
    detector.load(args.model_dir)
    result = detector.predict(image_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem
    heatmap_path = output_dir / f"{stem}_heatmap.jpg"
    overlay_path = output_dir / f"{stem}_overlay.jpg"

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

    print("\n========== PatchCore 预测结果 ==========")
    print(f"图片: {image_path}")
    print(f"Anomaly score: {result['anomaly_score']:.6f}")
    print(f"候选异常框: {bbox}")
    print(f"热力图: {heatmap_path.resolve()}")
    print(f"叠加图: {overlay_path.resolve()}")
    print("说明：当前版本尚未标定 PASS/NG 阈值。")
    print("========================================")


if __name__ == "__main__":
    main()
