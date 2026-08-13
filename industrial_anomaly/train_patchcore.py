from __future__ import annotations

import argparse
import sys
from pathlib import Path


CLEAN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CLEAN_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.anomaly.image_dataset import create_normal_dataloader
from app.anomaly.patchcore_adapter import PatchCoreAdapter, PatchCoreConfig


def parse_args():
    p = argparse.ArgumentParser(
        description="Train one product's PatchCore model using NORMAL images only."
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. bottle")
    p.add_argument(
        "--normal-dir",
        default=None,
        help="Override normal-image folder. Default: products/<product>/train/good",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Override model folder. Default: products/<product>/models/patchcore",
    )
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--coreset", type=float, default=0.10)
    p.add_argument("--imagesize", type=int, default=320)
    p.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Default keeps 256/224 ratio: 320 -> 366.",
    )
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    return p.parse_args()


def main():
    args = parse_args()
    product_dir = CLEAN_ROOT / "products" / args.product
    normal_dir = Path(args.normal_dir) if args.normal_dir else product_dir / "train" / "good"
    model_dir = Path(args.model_dir) if args.model_dir else product_dir / "models" / "patchcore"

    if not normal_dir.exists():
        raise FileNotFoundError(
            f"Normal training directory not found: {normal_dir.resolve()}\n"
            "PatchCore training must contain NORMAL images only."
        )
    if args.imagesize <= 0:
        raise ValueError("--imagesize must be > 0")

    resize = args.resize
    if resize is None:
        resize = max(args.imagesize, int(round(args.imagesize * (256.0 / 224.0))))
    if resize < args.imagesize:
        raise ValueError("--resize must be >= --imagesize")

    valid_layers = {"layer1", "layer2", "layer3", "layer4"}
    invalid = [layer for layer in args.layers if layer not in valid_layers]
    if invalid:
        raise ValueError(f"Unsupported layers: {invalid}")

    config = PatchCoreConfig(
        layers=tuple(args.layers),
        resize=resize,
        imagesize=args.imagesize,
        coreset_sampling_ratio=args.coreset,
    )
    loader = create_normal_dataloader(
        image_dir=normal_dir,
        batch_size=args.batch_size,
        num_workers=0,
        resize=config.resize,
        imagesize=config.imagesize,
    )
    detector = PatchCoreAdapter(device=args.device, config=config)

    print("========== Clean PatchCore Training ==========")
    print(f"product:       {args.product}")
    print(f"normal dir:    {normal_dir.resolve()}")
    print(f"normal images: {len(loader.dataset)}")
    print(f"resize/crop:   {config.resize}/{config.imagesize}")
    print(f"layers:        {config.layers}")
    print(f"coreset:       {config.coreset_sampling_ratio}")
    print(f"model dir:     {model_dir.resolve()}")
    print("NOTE: no defect labels/images are used in PatchCore training.")
    print("==============================================")

    detector.fit(loader)
    detector.save(model_dir)

    print("\nPatchCore training finished.")
    print(f"model: {model_dir.resolve()}")


if __name__ == "__main__":
    main()
