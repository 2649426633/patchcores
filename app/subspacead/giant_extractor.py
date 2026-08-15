from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel


@dataclass
class DINOv2GiantIndustrialConfig:
    model_ckpt: str = "weights/subspacead/dinov2-with-registers-giant"
    image_size: int = 672
    layers: tuple[int, ...] = (-12, -13, -14, -15, -16, -17, -18)
    aggregation: str = "mean"


class DINOv2GiantIndustrialExtractor:
    """DINOv2-Giant feature extractor for the industrial SubspaceAD path.

    The vendored official SubspaceAD extractor always asks the backbone for all
    attention matrices because its optional saliency-mask path needs them. The
    industrial tiled detector does not use that saliency mask, so computing the
    attentions wastes a large amount of time and VRAM at 672 px.

    This adapter keeps the exact same Hugging Face DINOv2-Giant backbone,
    preprocessing, hidden-state layers and mean aggregation, but requests hidden
    states only. The official code under third_party/subspacead remains untouched.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[DINOv2GiantIndustrialConfig] = None,
    ) -> None:
        self.config = config or DINOv2GiantIndustrialConfig()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        self.device = torch.device(device)
        self.processor = None
        self.model = None

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_model_ckpt(self) -> str:
        raw = Path(self.config.model_ckpt).expanduser()
        if raw.is_absolute():
            return str(raw.resolve())
        local = (self.project_root / raw).resolve()
        return str(local) if local.exists() else self.config.model_ckpt

    def load(self) -> None:
        model_ckpt = self.resolve_model_ckpt()
        print(f"[SubspaceAD/Giant-Industrial] device: {self.device}")
        print(f"[SubspaceAD/Giant-Industrial] model: {model_ckpt}")
        self.processor = AutoImageProcessor.from_pretrained(model_ckpt)
        self.model = AutoModel.from_pretrained(model_ckpt).eval().to(self.device)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        print("[SubspaceAD/Giant-Industrial] frozen backbone loaded (attentions disabled).")

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

    @torch.inference_mode()
    def extract_tokens(
        self,
        images: Sequence[str | Path | Image.Image],
    ) -> np.ndarray:
        """Return mean-aggregated patch tokens as [B,H,W,C]."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Extractor is not loaded. Call load() first.")
        if not images:
            raise ValueError("images must not be empty")

        pil_images = [self._as_pil(image) for image in images]
        res = int(self.config.image_size)
        inputs = self.processor(
            images=pil_images,
            return_tensors="pt",
            do_resize=True,
            size={"height": res, "width": res},
            do_center_crop=False,
            crop_size={"height": res, "width": res},
        ).to(self.device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Backbone did not return hidden states")

        cfg = self.model.config
        patch_size = int(cfg.patch_size)
        num_register_tokens = int(getattr(cfg, "num_register_tokens", 0))
        drop_front = 1 + num_register_tokens
        h_p = res // patch_size
        w_p = res // patch_size
        n_expected = h_p * w_p

        selected = []
        for layer in self.config.layers:
            state = hidden_states[layer]
            patch_tokens = state[:, drop_front : drop_front + n_expected, :]
            if patch_tokens.shape[1] != n_expected:
                raise RuntimeError(
                    f"Layer {layer}: expected {n_expected} patch tokens, "
                    f"got {patch_tokens.shape[1]}"
                )
            selected.append(patch_tokens)

        if self.config.aggregation != "mean":
            raise ValueError("Industrial Giant extractor currently supports mean aggregation only")
        fused = torch.stack(selected, dim=0).mean(dim=0)
        batch = fused.shape[0]
        feature_dim = fused.shape[-1]
        fused = fused.reshape(batch, h_p, w_p, feature_dim)
        return fused.detach().cpu().numpy().astype(np.float32)
