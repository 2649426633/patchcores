from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import pickle
from typing import Optional

import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models

import patchcore.common
import patchcore.patchcore
import patchcore.sampler


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
ADAPTER_CONFIG_FILENAME = "patchcore_adapter_config.json"


@dataclass
class PatchCoreConfig:
    backbone_name: str = "wideresnet50"
    layers: tuple[str, ...] = ("layer2", "layer3")
    resize: int = 256
    imagesize: int = 224
    pretrain_embed_dimension: int = 1024
    target_embed_dimension: int = 1024
    patchsize: int = 3
    patchstride: int = 1
    anomaly_score_num_nn: int = 1
    coreset_sampling_ratio: float = 0.1
    faiss_on_gpu: bool = False
    faiss_num_workers: int = 4


class PatchCoreAdapter:
    """Application adapter around the vendored PatchCore implementation.

    The adapter keeps the official PatchCore core files unchanged while making
    training and inference fully offline. WideResNet50-2 weights are loaded
    from ``weights/wide_resnet50_2-95faca4d.pth``.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        config: Optional[PatchCoreConfig] = None,
    ):
        self.config = config or PatchCoreConfig()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but the current PyTorch environment "
                "does not have an available CUDA device."
            )

        self.device = torch.device(device)
        self.model: Optional[patchcore.patchcore.PatchCore] = None

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    @property
    def local_backbone_path(self) -> Path:
        return self.project_root / "weights" / "wide_resnet50_2-95faca4d.pth"

    def _load_local_backbone(self):
        weight_path = self.local_backbone_path

        if not weight_path.exists():
            raise FileNotFoundError(
                "Missing local WideResNet50-2 weights:\n"
                f"{weight_path}\n\n"
                "Place wide_resnet50_2-95faca4d.pth in the project's "
                "weights directory."
            )

        print(f"[PatchCore] Loading local backbone: {weight_path}")

        backbone = models.wide_resnet50_2(weights=None)
        state_dict = torch.load(weight_path, map_location="cpu")

        if (
            isinstance(state_dict, dict)
            and "state_dict" in state_dict
            and isinstance(state_dict["state_dict"], dict)
        ):
            state_dict = state_dict["state_dict"]

        backbone.load_state_dict(state_dict, strict=True)
        backbone.name = self.config.backbone_name

        print("[PatchCore] Local WideResNet50-2 weights loaded.")
        return backbone

    def _build_nn_method(self):
        return patchcore.common.FaissNN(
            on_gpu=self.config.faiss_on_gpu,
            num_workers=self.config.faiss_num_workers,
        )

    def _new_model(self) -> patchcore.patchcore.PatchCore:
        cfg = self.config
        backbone = self._load_local_backbone()

        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            percentage=cfg.coreset_sampling_ratio,
            device=self.device,
        )

        model = patchcore.patchcore.PatchCore(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=list(cfg.layers),
            device=self.device,
            input_shape=(3, cfg.imagesize, cfg.imagesize),
            pretrain_embed_dimension=cfg.pretrain_embed_dimension,
            target_embed_dimension=cfg.target_embed_dimension,
            patchsize=cfg.patchsize,
            patchstride=cfg.patchstride,
            anomaly_score_num_nn=cfg.anomaly_score_num_nn,
            featuresampler=sampler,
            nn_method=self._build_nn_method(),
        )
        return model

    def fit(self, training_data) -> None:
        print(f"[PatchCore] device: {self.device}")
        self.model = self._new_model()
        self.model.fit(training_data)

    def save(self, model_dir: str | Path) -> Path:
        if self.model is None:
            raise RuntimeError("Model is not fitted or loaded; cannot save.")

        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_to_path(str(model_dir))

        adapter_config = {
            "format_version": 1,
            "resize": int(self.config.resize),
            "imagesize": int(self.config.imagesize),
            "layers": list(self.config.layers),
        }
        with open(model_dir / ADAPTER_CONFIG_FILENAME, "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, ensure_ascii=False, indent=2)

        print(f"[PatchCore] Model saved: {model_dir.resolve()}")
        return model_dir

    def load(self, model_dir: str | Path) -> None:
        """Load PatchCore fully offline without calling backbones.load()."""
        model_dir = Path(model_dir)
        params_file = model_dir / "patchcore_params.pkl"
        index_file = model_dir / "nnscorer_search_index.faiss"
        adapter_config_file = model_dir / ADAPTER_CONFIG_FILENAME

        missing = [p for p in (params_file, index_file) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Incomplete PatchCore model directory. Missing:\n"
                + "\n".join(str(p) for p in missing)
            )

        print(f"[PatchCore] Loading model directory: {model_dir.resolve()}")

        with open(params_file, "rb") as f:
            saved = pickle.load(f)

        backbone = self._load_local_backbone()
        backbone.name = saved.get("backbone.name", self.config.backbone_name)

        model = patchcore.patchcore.PatchCore(self.device)
        model.load(
            backbone=backbone,
            layers_to_extract_from=saved["layers_to_extract_from"],
            device=self.device,
            input_shape=saved["input_shape"],
            pretrain_embed_dimension=saved["pretrain_embed_dimension"],
            target_embed_dimension=saved["target_embed_dimension"],
            patchsize=saved["patchsize"],
            patchstride=saved["patchstride"],
            anomaly_score_num_nn=saved["anomaly_scorer_num_nn"],
            nn_method=self._build_nn_method(),
        )
        model.anomaly_scorer.load(str(model_dir))
        self.model = model

        loaded_h = int(self.model.input_shape[-2])
        loaded_w = int(self.model.input_shape[-1])
        if loaded_h != loaded_w:
            raise RuntimeError(
                "The current application adapter supports square PatchCore "
                f"inputs only, but the saved model uses {loaded_h}x{loaded_w}."
            )

        self.config.imagesize = loaded_h
        self.config.layers = tuple(saved["layers_to_extract_from"])

        if adapter_config_file.exists():
            with open(adapter_config_file, "r", encoding="utf-8") as f:
                adapter_config = json.load(f)

            saved_imagesize = int(adapter_config.get("imagesize", loaded_h))
            if saved_imagesize != loaded_h:
                raise RuntimeError(
                    "Adapter preprocessing metadata does not match PatchCore input shape: "
                    f"metadata imagesize={saved_imagesize}, model imagesize={loaded_h}."
                )

            self.config.resize = int(adapter_config["resize"])
            if "layers" in adapter_config:
                self.config.layers = tuple(adapter_config["layers"])
            metadata_source = ADAPTER_CONFIG_FILENAME
        else:
            # Backward compatibility for models saved before exact preprocessing
            # metadata was persisted. Legacy project models used the 256/224 ratio.
            self.config.resize = max(
                loaded_h,
                int(round(loaded_h * (256.0 / 224.0))),
            )
            metadata_source = "legacy inferred resize"

        print(
            f"[PatchCore] Preprocessing restored: "
            f"resize={self.config.resize}, imagesize={self.config.imagesize} "
            f"({metadata_source})"
        )
        print("[PatchCore] FAISS memory bank loaded.")
        print("[PatchCore] Offline model loaded.")

    def _image_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(self.config.resize),
                transforms.CenterCrop(self.config.imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def _load_image_tensor(self, image_path: str | Path) -> torch.Tensor:
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        tensor = self._image_transform()(image)
        return tensor.unsqueeze(0)

    def predict(self, image) -> dict:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load() or fit() first.")

        image_path = None

        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_tensor = self._load_image_tensor(image)
        elif isinstance(image, Image.Image):
            image_tensor = self._image_transform()(image.convert("RGB")).unsqueeze(0)
        elif torch.is_tensor(image):
            image_tensor = image
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
        else:
            raise TypeError(
                "predict() expects an image path, PIL.Image, or torch.Tensor."
            )

        scores, masks = self.model.predict(image_tensor)

        if len(scores) != 1 or len(masks) != 1:
            raise RuntimeError(
                "Single-image prediction returned an unexpected number of "
                f"results: scores={len(scores)}, masks={len(masks)}"
            )

        return {
            "image_path": image_path,
            "anomaly_score": float(np.asarray(scores[0]).item()),
            "anomaly_map": np.asarray(masks[0], dtype=np.float32),
        }
