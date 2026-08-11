from pathlib import Path
import pickle

import torch
import torchvision.models as models

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler


class PatchCoreAdapter:
    """Project adapter for offline PatchCore loading."""

    def __init__(self, device="cpu", config=None):
        self.device = torch.device(device)
        self.config = config
        self.model = None

    def _load_local_backbone(self):
        root = Path(__file__).resolve().parents[2]
        weight = root / "weights" / "wide_resnet50_2-95faca4d.pth"

        if not weight.exists():
            raise FileNotFoundError(f"Missing local backbone: {weight}")

        print(f"[PatchCore] Loading local backbone: {weight}")
        backbone = models.wide_resnet50_2(weights=None)
        state = torch.load(weight, map_location="cpu")
        backbone.load_state_dict(state, strict=True)
        backbone.name = "wideresnet50"
        return backbone

    def load(self, model_dir):
        model_dir = Path(model_dir)
        params = model_dir / "patchcore_params.pkl"

        with open(params, "rb") as f:
            cfg = pickle.load(f)

        backbone = self._load_local_backbone()

        nn_method = patchcore.common.FaissNN(
            on_gpu=False,
            num_workers=4,
        )

        self.model = patchcore.patchcore.PatchCore(self.device)
        self.model.load(
            backbone=backbone,
            layers_to_extract_from=cfg["layers_to_extract_from"],
            device=self.device,
            input_shape=cfg["input_shape"],
            pretrain_embed_dimension=cfg["pretrain_embed_dimension"],
            target_embed_dimension=cfg["target_embed_dimension"],
            patchsize=cfg["patchsize"],
            patchstride=cfg["patchstride"],
            anomaly_score_num_nn=cfg["anomaly_scorer_num_nn"],
            nn_method=nn_method,
        )

        self.model.anomaly_scorer.load(str(model_dir))
        print("[PatchCore] Offline model loaded.")

    def predict(self, image):
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return self.model.predict(image)
