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
    return parser.parse_args()


def main():
    args = parse_args()
    normal_dir = Path(args.normal_dir)
    if not normal_dir.exists():
        raise FileNotFoundError(
            f"Normal training directory not found: {normal_dir.resolve()}"
        )

    config = PatchCoreConfig(coreset_sampling_ratio=args.coreset)
    detector = PatchCoreAdapter(device=args.device, config=config)

    loader = create_normal_dataloader(
        image_dir=normal_dir,
        batch_size=args.batch_size,
        num_workers=0,
        resize=config.resize,
        imagesize=config.imagesize,
    )

    print(f"[PatchCore] 正常图片数量: {len(loader.dataset)}")
    print("[PatchCore] 正在建立 Memory Bank ...")
    detector.fit(loader)
    detector.save(args.model_dir)

    model_dir = Path(args.model_dir)
    print("\n训练完成。")
    print(f"参数文件: {model_dir / 'patchcore_params.pkl'}")
    print(f"FAISS索引: {model_dir / 'nnscorer_search_index.faiss'}")


if __name__ == "__main__":
    main()
