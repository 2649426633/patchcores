from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .preprocessing import build_patchcore_transform

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class NormalImageDataset(Dataset):
    """Dataset for a folder containing normal training images only."""

    def __init__(self, image_dir, resize=256, imagesize=224, recursive=False):
        self.image_dir = Path(image_dir)
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Normal image directory not found: {self.image_dir}")
        if not self.image_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.image_dir}")

        pattern = "**/*" if recursive else "*"
        self.image_paths = sorted(
            p
            for p in self.image_dir.glob(pattern)
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No supported images found in: {self.image_dir}")

        self.transform = build_patchcore_transform(resize=resize, imagesize=imagesize)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        return {"image": self.transform(image), "image_path": str(path)}


def create_normal_dataloader(
    image_dir,
    batch_size=8,
    num_workers=0,
    resize=256,
    imagesize=224,
    recursive=False,
):
    dataset = NormalImageDataset(
        image_dir=image_dir,
        resize=resize,
        imagesize=imagesize,
        recursive=recursive,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
