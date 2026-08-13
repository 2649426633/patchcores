from pathlib import Path

from PIL import Image
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


def map_display_bbox_to_original(
    image_path,
    bbox: tuple[int, int, int, int],
    resize=256,
    imagesize=224,
) -> tuple[int, int, int, int]:
    """Map a bbox from PatchCore's Resize->CenterCrop image back to the original image.

    ``bbox`` is expressed in the ``imagesize x imagesize`` display/crop coordinate
    system used by PatchCore localization. This function reproduces torchvision's
    resize geometry, restores the center-crop offset, and scales coordinates back
    to the source image. The returned box is clipped to the original image.
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    original = Image.open(image_path).convert("RGB")
    original_w, original_h = original.size
    resized = transforms.Resize(resize)(original)
    resized_w, resized_h = resized.size

    # torchvision CenterCrop uses rounded center offsets.
    crop_left = int(round((resized_w - imagesize) / 2.0))
    crop_top = int(round((resized_h - imagesize) / 2.0))

    x1, y1, x2, y2 = bbox
    rx1 = x1 + crop_left
    ry1 = y1 + crop_top
    rx2 = x2 + crop_left
    ry2 = y2 + crop_top

    sx = original_w / max(1.0, float(resized_w))
    sy = original_h / max(1.0, float(resized_h))

    ox1 = int(round(rx1 * sx))
    oy1 = int(round(ry1 * sy))
    ox2 = int(round(rx2 * sx))
    oy2 = int(round(ry2 * sy))

    ox1 = max(0, min(original_w - 1, ox1))
    oy1 = max(0, min(original_h - 1, oy1))
    ox2 = max(ox1 + 1, min(original_w, ox2))
    oy2 = max(oy1 + 1, min(original_h, oy2))
    return ox1, oy1, ox2, oy2
