from pathlib import Path

from PIL import Image
import torch
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_patchcore_transform(resize=256, imagesize=224):
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_image_tensor(image_path, resize=256, imagesize=224):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    return build_patchcore_transform(resize, imagesize)(image).unsqueeze(0)


def load_display_image(image_path, resize=256, imagesize=224):
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    display_transform = transforms.Compose(
        [transforms.Resize(resize), transforms.CenterCrop(imagesize)]
    )
    return display_transform(image)
