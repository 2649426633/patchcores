from __future__ import annotations

import argparse
from pathlib import Path

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import DINOv2Adapter


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Build a few-shot DINOv2 defect exemplar bank")
    parser.add_argument("--samples-dir", default="data/screw/test")
    parser.add_argument("--bank-dir", default="products/screw/defects/bank_3shot")
    parser.add_argument("--shots", type=int, default=3)
    parser.add_argument("--device", default=None, help="cpu, cuda or cuda:0")
    return parser.parse_args()


def image_files(folder: Path):
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def main():
    args = parse_args()
    samples_dir = Path(args.samples_dir)
    bank_dir = Path(args.bank_dir)

    if args.shots <= 0:
        raise ValueError("--shots must be > 0")
    if not samples_dir.exists():
        raise FileNotFoundError(samples_dir)

    class_dirs = sorted(
        [p for p in samples_dir.iterdir() if p.is_dir() and p.name.lower() != "good"],
        key=lambda p: p.name.lower(),
    )
    if not class_dirs:
        raise RuntimeError(f"No defect class directories found in {samples_dir}")

    extractor = DINOv2Adapter(device=args.device)
    extractor.load()

    embeddings = []
    labels = []
    support_paths = []

    print("\n========== DINOv2 few-shot support ==========")
    for class_dir in class_dirs:
        images = image_files(class_dir)
        if len(images) < args.shots:
            raise RuntimeError(
                f"Class {class_dir.name} has {len(images)} images, fewer than shots={args.shots}"
            )

        chosen = images[: args.shots]
        print(f"{class_dir.name}: {len(chosen)} support images")
        for image_path in chosen:
            embedding = extractor.embed(image_path)
            embeddings.append(embedding)
            labels.append(class_dir.name)
            support_paths.append(str(image_path.resolve()))
            print(f"  + {image_path.name}")

    import numpy as np

    bank = DefectExemplarBank(np.stack(embeddings, axis=0), labels, support_paths)
    bank.save(bank_dir)

    print("\n========== Defect bank built ==========")
    print(f"classes: {bank.classes}")
    print(f"num classes: {len(bank.classes)}")
    print(f"shots/class: {args.shots}")
    print(f"num exemplars: {len(bank.labels)}")
    print(f"embedding shape: {bank.embeddings.shape}")
    print(f"bank dir: {bank_dir.resolve()}")
    print("=======================================")


if __name__ == "__main__":
    main()
