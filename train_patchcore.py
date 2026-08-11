import argparse
from pathlib import Path

from app.anomaly.image_dataset import create_normal_dataloader
from app.anomaly.patchcore_adapter import PatchCoreAdapter, PatchCoreConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Train PatchCore on normal images")
    parser.add_argument(
        "--normal-dir",
        default="data/screw/train/good",
        help="Normal training image directory",
    )
    parser.add_argument(
        "--model-dir",
        default="products/screw/models/patchcore",
        help="Directory used to save PatchCore model files",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument(
        "--imagesize",
        type=int,
        default=224,
        help="PatchCore square input/crop size. Baseline: 224; high-resolution experiment: 320.",
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help=(
            "Resize shorter side before center crop. If omitted, keeps the baseline "
            "256/224 ratio automatically (224->256, 320->366)."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["layer2", "layer3"],
        help=(
            "Backbone feature layers used by PatchCore. Baseline: layer2 layer3. "
            "For finer texture experiments use: --layers layer1 layer2"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    normal_dir = Path(args.normal_dir)
    if not normal_dir.exists():
        raise FileNotFoundError(
            f"Normal training directory not found: {normal_dir.resolve()}"
        )

    if args.imagesize <= 0:
        raise ValueError("--imagesize must be > 0")

    resize = args.resize
    if resize is None:
        resize = max(
            args.imagesize,
            int(round(args.imagesize * (256.0 / 224.0))),
        )

    if resize < args.imagesize:
        raise ValueError(
            f"--resize ({resize}) must be >= --imagesize ({args.imagesize})"
        )

    valid_layers = {"layer1", "layer2", "layer3", "layer4"}
    invalid_layers = [layer for layer in args.layers if layer not in valid_layers]
    if invalid_layers:
        raise ValueError(
            f"Unsupported --layers: {invalid_layers}. Valid choices: {sorted(valid_layers)}"
        )

    config = PatchCoreConfig(
        layers=tuple(args.layers),
        resize=resize,
        imagesize=args.imagesize,
        coreset_sampling_ratio=args.coreset,
    )
    detector = PatchCoreAdapter(device=args.device, config=config)

    loader = create_normal_dataloader(
        image_dir=normal_dir,
        batch_size=args.batch_size,
        num_workers=0,
        resize=config.resize,
        imagesize=config.imagesize,
    )

    print("========== PatchCore 训练配置 ==========")
    print(f"正常图片目录: {normal_dir.resolve()}")
    print(f"正常图片数量: {len(loader.dataset)}")
    print(f"resize: {config.resize}")
    print(f"imagesize: {config.imagesize}")
    print(f"layers: {config.layers}")
    print(f"coreset: {config.coreset_sampling_ratio}")
    print(f"模型输出: {Path(args.model_dir).resolve()}")
    print("======================================")

    print("[PatchCore] 正在建立 Memory Bank ...")
    detector.fit(loader)
    detector.save(args.model_dir)

    model_dir = Path(args.model_dir)
    print("\n训练完成。")
    print(f"参数文件: {model_dir / 'patchcore_params.pkl'}")
    print(f"FAISS索引: {model_dir / 'nnscorer_search_index.faiss'}")


if __name__ == "__main__":
    main()
