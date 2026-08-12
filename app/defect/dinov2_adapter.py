from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
FEATURE_MODES = ("cls", "patch_mean", "patch_center")


@dataclass
class DINOv2Config:
    model_name: str = "dinov2_vits14"
    image_size: int = 224
    embedding_dim: int = 384
    repo_dir: str = "third_party/dinov2"
    weights_path: str = "weights/dinov2_vits14_pretrain.pth"
    center_fraction: float = 0.50


class DINOv2Adapter:
    """Frozen DINOv2 feature extractor for few-shot defect recognition.

    The engineering default remains the DINOv2 ViT-S/14 CLS token. For local
    industrial defects, ``patch_mean`` and ``patch_center`` expose alternative
    embeddings derived from DINOv2's normalized patch tokens without changing
    or fine-tuning the backbone.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[DINOv2Config] = None,
    ):
        self.config = config or DINOv2Config()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        self.device = torch.device(device)
        self.model: Optional[torch.nn.Module] = None

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def repo_dir(self) -> Path:
        path = Path(self.config.repo_dir)
        return path if path.is_absolute() else self.project_root / path

    @property
    def weights_path(self) -> Path:
        path = Path(self.config.weights_path)
        return path if path.is_absolute() else self.project_root / path

    def load(self) -> None:
        repo_dir = self.repo_dir
        weights_path = self.weights_path

        if not repo_dir.exists():
            raise FileNotFoundError(
                "Local DINOv2 repository not found:\n"
                f"{repo_dir}\n"
                "Clone facebookresearch/dinov2 into this directory first."
            )
        if not (repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(
                f"hubconf.py not found in local DINOv2 repository: {repo_dir}"
            )
        if not weights_path.exists():
            raise FileNotFoundError(
                "Local DINOv2 weights not found:\n"
                f"{weights_path}\n"
                "Expected official dinov2_vits14_pretrain.pth weights."
            )

        print(f"[DINOv2] device: {self.device}")
        print(f"[DINOv2] local repo: {repo_dir}")
        print(f"[DINOv2] local weights: {weights_path}")

        model = torch.hub.load(
            str(repo_dir),
            self.config.model_name,
            source="local",
            pretrained=True,
            weights=str(weights_path),
        )
        model.eval()
        model.to(self.device)

        for parameter in model.parameters():
            parameter.requires_grad_(False)

        self.model = model
        print("[DINOv2] frozen backbone loaded.")

    def _transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _prepare_image(self, image: str | Path | Image.Image) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            image_path = Path(image)
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            pil_image = Image.open(image_path).convert("RGB")
        elif isinstance(image, Image.Image):
            pil_image = image.convert("RGB")
        else:
            raise TypeError("image must be a path or PIL.Image")

        tensor = self._transform()(pil_image)
        return tensor.unsqueeze(0).to(self.device)

    def _pool_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        feature_mode: str,
        center_fraction: float,
    ) -> torch.Tensor:
        if patch_tokens.ndim != 3 or patch_tokens.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected patch-token shape: {tuple(patch_tokens.shape)}"
            )

        if feature_mode == "patch_mean":
            return patch_tokens.mean(dim=1)

        if feature_mode != "patch_center":
            raise ValueError(f"Unsupported patch feature mode: {feature_mode}")

        n_tokens = int(patch_tokens.shape[1])
        grid = int(round(math.sqrt(n_tokens)))
        if grid * grid != n_tokens:
            raise RuntimeError(
                "patch_center expects a square token grid, but got "
                f"{n_tokens} patch tokens"
            )

        fraction = float(center_fraction)
        if not (0.0 < fraction <= 1.0):
            raise ValueError("center_fraction must be in (0, 1]")

        keep = max(1, min(grid, int(round(grid * fraction))))
        start = (grid - keep) // 2
        end = start + keep

        grid_tokens = patch_tokens.reshape(1, grid, grid, patch_tokens.shape[-1])
        center_tokens = grid_tokens[:, start:end, start:end, :]
        return center_tokens.reshape(1, -1, patch_tokens.shape[-1]).mean(dim=1)

    @torch.inference_mode()
    def embed(
        self,
        image: str | Path | Image.Image,
        feature_mode: str = "cls",
        center_fraction: Optional[float] = None,
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("DINOv2 is not loaded. Call load() first.")

        feature_mode = str(feature_mode).strip().lower()
        if feature_mode not in FEATURE_MODES:
            raise ValueError(
                f"feature_mode must be one of {FEATURE_MODES}, got {feature_mode!r}"
            )

        tensor = self._prepare_image(image)
        features = self.model.forward_features(tensor)

        if feature_mode == "cls":
            embedding = features["x_norm_clstoken"]
        else:
            patch_tokens = features["x_norm_patchtokens"]
            embedding = self._pool_patch_tokens(
                patch_tokens,
                feature_mode=feature_mode,
                center_fraction=(
                    self.config.center_fraction
                    if center_fraction is None
                    else float(center_fraction)
                ),
            )

        embedding = F.normalize(embedding, p=2, dim=-1)

        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise RuntimeError(
                f"Unexpected DINOv2 embedding shape: {tuple(embedding.shape)}"
            )
        if embedding.shape[1] != self.config.embedding_dim:
            raise RuntimeError(
                "Unexpected embedding dimension: "
                f"got {embedding.shape[1]}, expected {self.config.embedding_dim}"
            )

        return embedding[0].detach().cpu().numpy().astype(np.float32)
