from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from app.defect.dinov2_adapter import DINOv2Adapter, DINOv2Config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load local frozen DINOv2 ViT-S/14 and extract one L2-normalized embedding."
    )
    parser.add_argument(
        "--image",
        default="data/screw/test/scratch_head/000.png",
        help="Image/ROI used for the smoke test.",
    )
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    parser.add_argument("--repo-dir", default="third_party/dinov2")
    parser.add_argument("--weights", default="weights/dinov2_vits14_pretrain.pth")
    parser.add_argument(
        "--save",
        default=None,
        help="Optional .npy output path for the extracted embedding.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    config = DINOv2Config(
        repo_dir=args.repo_dir,
        weights_path=args.weights,
    )
    extractor = DINOv2Adapter(device=args.device, config=config)
    extractor.load()

    embedding = extractor.embed(args.image)

    print("========== DINOv2 embedding smoke test ==========")
    print(f"image: {Path(args.image).resolve()}")
    print(f"shape: {embedding.shape}")
    print(f"dtype: {embedding.dtype}")
    print(f"L2 norm: {np.linalg.norm(embedding):.6f}")
    print(f"first 8 values: {np.array2string(embedding[:8], precision=6)}")
    print("=================================================")

    if args.save:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embedding)
        print(f"saved: {output_path.resolve()}")


if __name__ == "__main__":
    main()
