from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class DINOv2SSubspaceConfig:
    """Industrial SubspaceAD backbone settings based on the paper's DINOv2-S ablation.

    The official SubspaceAD backbone-ablation script uses DINOv2-S with 448x448
    input and Hugging Face hidden-state indices -4,-5. For the official
    facebookresearch/dinov2 ViT-S/14 implementation, those correspond to the
    outputs of transformer blocks 7 and 8 when the embedding state is counted
    as hidden-state 0.
    """

    model_name: str = "dinov2_vits14"
    image_size: int = 448
    patch_size: int = 14
    feature_dim: int = 384
    block_indices: tuple[int, int] = (7, 8)
    repo_dir: str = "third_party/dinov2"
    weights_path: str = "weights/dinov2_vits14_pretrain.pth"


class DINOv2SSubspaceExtractor:
    """Frozen local DINOv2-S/14 patch extractor for SubspaceAD.

    This class deliberately does not use Hugging Face and does not modify the
    vendored official SubspaceAD code. It loads the already-used local
    facebookresearch/dinov2 repository and .pth weights, then mean-pools the
    selected intermediate transformer-layer patch tokens.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[DINOv2SSubspaceConfig] = None,
    ) -> None:
        self.config = config or DINOv2SSubspaceConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
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

    @property
    def grid_size(self) -> int:
        if self.config.image_size % self.config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        return self.config.image_size // self.config.patch_size

    def load(self) -> None:
        if not self.repo_dir.exists():
            raise FileNotFoundError(
                f"Local DINOv2 repository not found: {self.repo_dir}\n"
                "Expected third_party/dinov2 with hubconf.py."
            )
        if not (self.repo_dir / "hubconf.py").exists():
            raise FileNotFoundError(f"hubconf.py not found: {self.repo_dir}")
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"DINOv2-S weights not found: {self.weights_path}\n"
                "Expected weights/dinov2_vits14_pretrain.pth."
            )

        print(f"[SubspaceAD/DINOv2-S] device: {self.device}")
        print(f"[SubspaceAD/DINOv2-S] repo: {self.repo_dir}")
        print(f"[SubspaceAD/DINOv2-S] weights: {self.weights_path}")

        model = torch.hub.load(
            str(self.repo_dir),
            self.config.model_name,
            source="local",
            pretrained=True,
            weights=str(self.weights_path),
        )
        model.eval().to(self.device)
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        if not hasattr(model, "get_intermediate_layers"):
            raise RuntimeError(
                "Loaded DINOv2 model does not expose get_intermediate_layers(). "
                "Check the local facebookresearch/dinov2 source version."
            )

        self.model = model
        print("[SubspaceAD/DINOv2-S] frozen backbone loaded.")

    def _transform(self):
        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    @staticmethod
    def _as_pil(image: str | Path | Image.Image) -> Image.Image:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                raise FileNotFoundError(path)
            return Image.open(path).convert("RGB")
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        raise TypeError("image must be a path or PIL.Image")

    def _prepare_batch(
        self, images: Sequence[str | Path | Image.Image]
    ) -> torch.Tensor:
        if not images:
            raise ValueError("images must not be empty")
        transform = self._transform()
        tensors = [transform(self._as_pil(image)) for image in images]
        return torch.stack(tensors, dim=0).to(self.device)

    @torch.inference_mode()
    def extract_tokens(
        self, images: Sequence[str | Path | Image.Image]
    ) -> np.ndarray:
        """Return mean-aggregated intermediate patch tokens as [B,H,W,384]."""
        if self.model is None:
            raise RuntimeError("DINOv2-S is not loaded. Call load() first.")

        tensor = self._prepare_batch(images)
        outputs = self.model.get_intermediate_layers(
            tensor,
            n=list(self.config.block_indices),
            reshape=False,
            return_class_token=False,
            norm=True,
        )
        if len(outputs) != len(self.config.block_indices):
            raise RuntimeError(
                f"Expected {len(self.config.block_indices)} intermediate layers, "
                f"got {len(outputs)}"
            )

        for index, output in zip(self.config.block_indices, outputs):
            if output.ndim != 3:
                raise RuntimeError(
                    f"Unexpected block {index} token shape: {tuple(output.shape)}"
                )
            if output.shape[-1] != self.config.feature_dim:
                raise RuntimeError(
                    f"Unexpected DINOv2-S dim {output.shape[-1]}, "
                    f"expected {self.config.feature_dim}"
                )

        fused = torch.stack(list(outputs), dim=0).mean(dim=0)
        expected_tokens = self.grid_size * self.grid_size
        if fused.shape[1] != expected_tokens:
            raise RuntimeError(
                f"Expected {expected_tokens} patch tokens at "
                f"{self.config.image_size}px, got {fused.shape[1]}"
            )

        b = fused.shape[0]
        fused = fused.reshape(
            b, self.grid_size, self.grid_size, self.config.feature_dim
        )
        return fused.detach().cpu().numpy().astype(np.float32)
