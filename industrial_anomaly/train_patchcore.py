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
from app.anomaly.tiled import create_tiled_normal_dataloader, save_inspection_config


def resolve_clean_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = CLEAN_ROOT / path
    return path.resolve()


def parse_args():
    p = argparse.ArgumentParser(
        description="Train one product's PatchCore model using NORMAL images only."
    )
    p.add_argument("--product", required=True, help="Product/SKU name, e.g. phone")
    p.add_argument("--normal-dir", default=None)
    p.add_argument("--model-dir", default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    p.add_argument("--coreset", type=float, default=0.10)
    p.add_argument("--imagesize", type=int, default=320)
    p.add_argument("--resize", type=int, default=None)
    p.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    p.add_argument(
        "--mode",
        choices=["center", "tiled"],
        default="center",
        help=(
            "center = legacy Resize+CenterCrop; tiled = full-image overlapping tiles. "
            "Use tiled for high-resolution industrial images with defects anywhere."
        ),
    )
    p.add_argument(
        "--tile-fraction",
        type=float,
        default=0.75,
        help="Tiled mode: square tile side as fraction of the shorter image side.",
    )
    p.add_argument(
        "--tile-overlap",
        type=float,
        default=0.25,
        help="Tiled mode: overlap fraction between neighboring tiles.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    product_dir = CLEAN_ROOT / "products" / args.product
    normal_dir = (
        resolve_clean_path(args.normal_dir)
        if args.normal_dir
        else product_dir / "train" / "good"
    )
    model_dir = (
        resolve_clean_path(args.model_dir)
        if args.model_dir
        else product_dir / "models" / "patchcore"
    )

    if not normal_dir.is_dir():
        raise FileNotFoundError(
            f"Normal training directory not found: {normal_dir}\n"
            "PatchCore training must contain NORMAL images only."
        )
    if args.imagesize <= 0:
        raise ValueError("--imagesize must be > 0")

    valid_layers = {"layer1", "layer2", "layer3", "layer4"}
    invalid = [layer for layer in args.layers if layer not in valid_layers]
    if invalid:
        raise ValueError(f"Unsupported layers: {invalid}")

    if args.mode == "tiled":
        # Tiled crops are already square and directly resized to imagesize.
        resize = args.imagesize
        config = PatchCoreConfig(
            layers=tuple(args.layers),
            resize=resize,
            imagesize=args.imagesize,
            coreset_sampling_ratio=args.coreset,
        )
        loader = create_tiled_normal_dataloader(
            image_dir=normal_dir,
            batch_size=args.batch_size,
            num_workers=0,
            imagesize=args.imagesize,
            tile_fraction=args.tile_fraction,
            overlap=args.tile_overlap,
        )
        source_images = len(loader.dataset.image_paths)
        training_views = len(loader.dataset)
    else:
        resize = args.resize
        if resize is None:
            resize = max(
                args.imagesize,
                int(round(args.imagesize * (256.0 / 224.0))),
            )
        if resize < args.imagesize:
            raise ValueError("--resize must be >= --imagesize")
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
        source_images = len(loader.dataset)
        training_views = source_images

    detector = PatchCoreAdapter(device=args.device, config=config)

    print("========== PatchCore Training ==========")
    print(f"clean root:       {CLEAN_ROOT}")
    print(f"repo root:        {REPO_ROOT}")
    print(f"product:          {args.product}")
    print(f"mode:             {args.mode}")
    print(f"normal dir:       {normal_dir}")
    print(f"source images:    {source_images}")
    print(f"training views:   {training_views}")
    if args.mode == "tiled":
        print(f"tile fraction:    {args.tile_fraction}")
        print(f"tile overlap:     {args.tile_overlap}")
        print(f"tile input:       {args.imagesize}x{args.imagesize} direct square resize")
    else:
        print(f"resize/crop:      {config.resize}/{config.imagesize}")
    print(f"layers:           {config.layers}")
    print(f"coreset:          {config.coreset_sampling_ratio}")
    print(f"model dir:        {model_dir}")
    print("NOTE: no defect labels/images are used in PatchCore training.")
    print("========================================")

    detector.fit(loader)
    detector.save(model_dir)
    save_inspection_config(
        model_dir,
        mode="tiled" if args.mode == "tiled" else "center_crop",
        imagesize=args.imagesize,
        tile_fraction=args.tile_fraction if args.mode == "tiled" else None,
        tile_overlap=args.tile_overlap if args.mode == "tiled" else None,
    )

    print("\nPatchCore training finished.")
    print(f"model: {model_dir.resolve()}")
    print(f"inspection mode: {args.mode}")


if __name__ == "__main__":
    main()
