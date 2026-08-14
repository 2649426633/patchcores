from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


DEPLOY_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEPLOY_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def log(message: str) -> None:
    print(message, flush=True)


class PatchCoreFeatureExport(nn.Module):
    """Generic PatchCore feature + nearest-neighbour scoring graph.

    images:      [B, 3, 320, 320]
    memory_bank: [M, 1024]

    returns:
      patch_embeddings: [B, 1600, 1024]
      patch_scores:     [B, 1600]
    """

    def __init__(
        self,
        backbone: nn.Module,
        input_size: int = 320,
        patch_size: int = 3,
        pretrain_dim: int = 1024,
        target_dim: int = 1024,
    ) -> None:
        super().__init__()
        if input_size % 16 != 0:
            raise ValueError("input_size must be divisible by 16")
        if patch_size % 2 != 1:
            raise ValueError("patch_size must be odd")

        self.backbone = backbone
        self.input_size = int(input_size)
        self.patch_size = int(patch_size)
        self.padding = int((patch_size - 1) // 2)
        self.pretrain_dim = int(pretrain_dim)
        self.target_dim = int(target_dim)
        self.layer2_grid = self.input_size // 8
        self.layer3_grid = self.input_size // 16

    def _backbone_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = self.backbone
        x = b.conv1(x)
        x = b.bn1(x)
        x = b.relu(x)
        x = b.maxpool(x)
        x = b.layer1(x)
        layer2 = b.layer2(x)
        layer3 = b.layer3(layer2)
        return layer2, layer3

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        return F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=1,
            padding=self.padding,
        ).transpose(1, 2)

    def _mean_map(self, x: torch.Tensor) -> torch.Tensor:
        batch, patches, dim = x.shape
        x = x.reshape(batch * patches, 1, dim)
        x = F.adaptive_avg_pool1d(x, self.pretrain_dim).squeeze(1)
        return x.reshape(batch, patches, self.pretrain_dim)

    def _embed(self, images: torch.Tensor) -> torch.Tensor:
        layer2, layer3 = self._backbone_features(images)

        patch2 = self._patchify(layer2)
        patch3 = self._patchify(layer3)

        b = patch3.shape[0]
        patch3_map = patch3.transpose(1, 2).reshape(
            b, patch3.shape[-1], self.layer3_grid, self.layer3_grid
        )
        patch3_map = F.interpolate(
            patch3_map,
            size=(self.layer2_grid, self.layer2_grid),
            mode="bilinear",
            align_corners=False,
        )
        patch3 = patch3_map.flatten(2).transpose(1, 2)

        mapped2 = self._mean_map(patch2)
        mapped3 = self._mean_map(patch3)

        merged = torch.stack((mapped2, mapped3), dim=2)
        b, n, layers, dim = merged.shape
        merged = merged.reshape(b * n, 1, layers * dim)
        merged = F.adaptive_avg_pool1d(merged, self.target_dim).squeeze(1)
        return merged.reshape(b, n, self.target_dim)

    def forward(
        self,
        images: torch.Tensor,
        memory_bank: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self._embed(images)

        query = embeddings.reshape(-1, self.target_dim)
        query_sq = (query * query).sum(dim=1, keepdim=True)
        memory_sq = (memory_bank * memory_bank).sum(dim=1).unsqueeze(0)
        squared_l2 = query_sq + memory_sq - 2.0 * (
            query @ memory_bank.transpose(0, 1)
        )
        squared_l2 = torch.clamp(squared_l2, min=0.0)
        patch_scores = torch.min(squared_l2, dim=1).values
        patch_scores = patch_scores.reshape(embeddings.shape[0], embeddings.shape[1])
        return embeddings, patch_scores


class DINOv2FeatureExport(nn.Module):
    """Frozen DINOv2 ViT-S/14 CLS + center-patch feature graph."""

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 224,
        patch_size: int = 14,
        embedding_dim: int = 384,
        center_fraction: float = 0.50,
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = int(image_size)
        self.patch_size = int(patch_size)
        self.embedding_dim = int(embedding_dim)
        self.grid = self.image_size // self.patch_size
        keep = max(1, min(self.grid, int(round(self.grid * center_fraction))))
        self.center_start = (self.grid - keep) // 2
        self.center_end = self.center_start + keep

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.forward_features(images)
        cls = features["x_norm_clstoken"]
        patch_tokens = features["x_norm_patchtokens"]

        grid_tokens = patch_tokens.reshape(
            patch_tokens.shape[0], self.grid, self.grid, self.embedding_dim
        )
        center = grid_tokens[
            :,
            self.center_start : self.center_end,
            self.center_start : self.center_end,
            :,
        ]
        center = center.reshape(center.shape[0], -1, self.embedding_dim).mean(dim=1)

        cls = F.normalize(cls, p=2, dim=-1)
        center = F.normalize(center, p=2, dim=-1)
        return cls, center


def load_patchcore_backbone(device: torch.device) -> nn.Module:
    weight_path = REPO_ROOT / "weights" / "wide_resnet50_2-95faca4d.pth"
    if not weight_path.exists():
        raise FileNotFoundError(f"Missing PatchCore backbone weights: {weight_path}")

    log("[1/4] Creating WideResNet50-2...")
    backbone = models.wide_resnet50_2(weights=None)
    log(f"[1/4] Loading PatchCore weights: {weight_path}")
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=True)
    log(f"[1/4] Moving PatchCore backbone to {device}...")
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    log("[1/4] PatchCore backbone ready.")
    return backbone


def load_dinov2(device: torch.device) -> nn.Module:
    repo_dir = REPO_ROOT / "third_party" / "dinov2"
    weights_path = REPO_ROOT / "weights" / "dinov2_vits14_pretrain.pth"
    if not (repo_dir / "hubconf.py").exists():
        raise FileNotFoundError(f"Missing local DINOv2 repository: {repo_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing DINOv2 weights: {weights_path}")

    log(f"[3/4] Loading DINOv2 from: {repo_dir}")
    log(f"[3/4] DINOv2 weights: {weights_path}")
    model = torch.hub.load(
        str(repo_dir),
        "dinov2_vits14",
        source="local",
        pretrained=True,
        weights=str(weights_path),
    )
    log(f"[3/4] Moving DINOv2 to {device}...")
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    log("[3/4] DINOv2 ready.")
    return model


def export_patchcore_legacy(
    model: PatchCoreFeatureExport,
    image_dummy: torch.Tensor,
    memory_dummy: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        (image_dummy, memory_dummy),
        str(output_path),
        input_names=["images", "memory_bank"],
        output_names=["patch_embeddings", "patch_scores"],
        opset_version=opset,
        export_params=True,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "memory_bank": {0: "memory_rows"},
            "patch_embeddings": {0: "batch"},
            "patch_scores": {0: "batch"},
        },
    )


def export_patchcore_dynamo(
    model: PatchCoreFeatureExport,
    image_dummy: torch.Tensor,
    memory_dummy: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        (image_dummy, memory_dummy),
        str(output_path),
        input_names=["images", "memory_bank"],
        output_names=["patch_embeddings", "patch_scores"],
        opset_version=opset,
        export_params=True,
        dynamo=True,
        optimize=False,
        dynamic_shapes={
            "images": {0: torch.export.Dim("batch", min=1)},
            "memory_bank": {0: torch.export.Dim("memory_rows", min=1)},
        },
    )


def export_patchcore(
    model: PatchCoreFeatureExport,
    image_dummy: torch.Tensor,
    memory_dummy: torch.Tensor,
    output_path: Path,
    opset: int,
    exporter: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    log(f"[2/4] Exporting PatchCore with {exporter} exporter...")

    if exporter == "legacy":
        export_patchcore_legacy(model, image_dummy, memory_dummy, output_path, opset)
    else:
        try:
            export_patchcore_dynamo(model, image_dummy, memory_dummy, output_path, opset)
        except Exception as exc:
            log(f"[2/4] Dynamo failed: {type(exc).__name__}: {exc}")
            log("[2/4] Falling back to legacy exporter...")
            export_patchcore_legacy(model, image_dummy, memory_dummy, output_path, opset)
            exporter = "legacy-fallback"

    log(f"[2/4] PatchCore exported ({exporter}): {output_path.resolve()}")


def export_dino_legacy(
    model: DINOv2FeatureExport,
    dummy: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["images"],
        output_names=["cls_embedding", "center_embedding"],
        opset_version=opset,
        export_params=True,
        dynamo=False,
        do_constant_folding=True,
        dynamic_axes={
            "images": {0: "batch"},
            "cls_embedding": {0: "batch"},
            "center_embedding": {0: "batch"},
        },
    )


def export_dino_dynamo(
    model: DINOv2FeatureExport,
    dummy: torch.Tensor,
    output_path: Path,
    opset: int,
) -> None:
    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["images"],
        output_names=["cls_embedding", "center_embedding"],
        opset_version=opset,
        export_params=True,
        dynamo=True,
        optimize=False,
        dynamic_shapes={"images": {0: torch.export.Dim("batch", min=1)}},
    )


def export_dino(
    model: DINOv2FeatureExport,
    dummy: torch.Tensor,
    output_path: Path,
    opset: int,
    exporter: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    log(f"[4/4] Exporting DINOv2 with {exporter} exporter...")

    if exporter == "legacy":
        export_dino_legacy(model, dummy, output_path, opset)
    else:
        try:
            export_dino_dynamo(model, dummy, output_path, opset)
        except Exception as exc:
            log(f"[4/4] Dynamo failed: {type(exc).__name__}: {exc}")
            log("[4/4] Falling back to legacy exporter...")
            export_dino_legacy(model, dummy, output_path, opset)
            exporter = "legacy-fallback"

    log(f"[4/4] DINOv2 exported ({exporter}): {output_path.resolve()}")


def verify_patchcore(
    model: PatchCoreFeatureExport,
    image_dummy: torch.Tensor,
    memory_dummy: torch.Tensor,
    onnx_path: Path,
) -> None:
    import onnx
    import onnxruntime as ort

    log("[verify] Checking PatchCore ONNX...")
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.inference_mode():
        pt_outputs = model(image_dummy, memory_dummy)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(
        None,
        {
            "images": image_dummy.detach().cpu().numpy(),
            "memory_bank": memory_dummy.detach().cpu().numpy(),
        },
    )
    for index, (pt, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        pt_np = pt.detach().cpu().numpy()
        if pt_np.shape != ort_out.shape:
            raise RuntimeError(
                f"PatchCore output {index} shape mismatch: {pt_np.shape} vs {ort_out.shape}"
            )
        log(
            f"[verify] PatchCore[{index}] shape={pt_np.shape} "
            f"max_abs={float(np.max(np.abs(pt_np - ort_out))):.6g}"
        )


def verify_dino(
    model: DINOv2FeatureExport,
    dummy: torch.Tensor,
    onnx_path: Path,
) -> None:
    import onnx
    import onnxruntime as ort

    log("[verify] Checking DINOv2 ONNX...")
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.inference_mode():
        pt_outputs = model(dummy)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"images": dummy.detach().cpu().numpy()})
    for index, (pt, ort_out) in enumerate(zip(pt_outputs, ort_outputs)):
        pt_np = pt.detach().cpu().numpy()
        if pt_np.shape != ort_out.shape:
            raise RuntimeError(
                f"DINOv2 output {index} shape mismatch: {pt_np.shape} vs {ort_out.shape}"
            )
        log(
            f"[verify] DINOv2[{index}] shape={pt_np.shape} "
            f"max_abs={float(np.max(np.abs(pt_np - ort_out))):.6g}"
        )


def write_engine_config(output_dir: Path) -> None:
    patch_path = output_dir / "patchcore_feature.onnx"
    dino_path = output_dir / "dinov2_feature.onnx"
    if not patch_path.exists() or not dino_path.exists():
        log("[config] Waiting for both ONNX files before writing engine_config.json.")
        return

    config = {
        "format_version": 2,
        "patchcore": {
            "file": patch_path.name,
            "input": "images",
            "memory_input": "memory_bank",
            "input_shape": [1, 3, 320, 320],
            "outputs": ["patch_embeddings", "patch_scores"],
            "output_shape": [1, 1600, 1024],
            "score_shape": [1, 1600],
            "score_metric": "min_squared_l2",
            "patch_grid": [40, 40],
            "embedding_dim": 1024,
            "layers": ["layer2", "layer3"],
            "patch_size": 3,
            "patch_stride": 1,
        },
        "dinov2": {
            "file": dino_path.name,
            "input": "images",
            "input_shape": [1, 3, 224, 224],
            "outputs": ["cls_embedding", "center_embedding"],
            "embedding_dim": 384,
            "model": "dinov2_vits14",
            "center_fraction": 0.50,
        },
        "normalization": {
            "mean": IMAGENET_MEAN,
            "std": IMAGENET_STD,
            "channel_order": "RGB",
            "tensor_layout": "NCHW",
        },
    }
    config_path = output_dir / "engine_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    log(f"[config] {config_path.resolve()}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export generic PatchCore + DINOv2 ONNX inference engine."
    )
    p.add_argument(
        "--output-dir",
        default=str(DEPLOY_ROOT / "models"),
        help="Output folder for generic ONNX files.",
    )
    p.add_argument(
        "--model",
        choices=["all", "patchcore", "dinov2"],
        default="all",
        help="Export both models or isolate one model for troubleshooting.",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Export device. CPU is recommended on Windows; runtime can still use CUDA.",
    )
    p.add_argument(
        "--exporter",
        choices=["legacy", "dynamo"],
        default="legacy",
        help="Legacy is the stable default for these fixed inference graphs on Windows.",
    )
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--skip-verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        log("[warning] CUDA requested but unavailable; using CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    log("========== Generic ONNX Export ==========")
    log(f"repo root:   {REPO_ROOT}")
    log(f"output dir:  {output_dir}")
    log(f"model:       {args.model}")
    log(f"device:      {device}")
    log(f"exporter:    {args.exporter}")
    log(f"opset:       {args.opset}")
    log("=========================================")

    if args.model in ("all", "patchcore"):
        patchcore = PatchCoreFeatureExport(load_patchcore_backbone(device)).to(device).eval()
        patch_dummy = torch.randn(1, 3, 320, 320, device=device, dtype=torch.float32)
        memory_dummy = torch.randn(32, 1024, device=device, dtype=torch.float32)
        patch_path = output_dir / "patchcore_feature.onnx"
        export_patchcore(
            patchcore,
            patch_dummy,
            memory_dummy,
            patch_path,
            args.opset,
            args.exporter,
        )
        if not args.skip_verify:
            log("[verify] Moving PatchCore verification to CPU...")
            verify_patchcore(
                patchcore.cpu(),
                patch_dummy.cpu(),
                memory_dummy.cpu(),
                patch_path,
            )
        del patchcore, patch_dummy, memory_dummy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in ("all", "dinov2"):
        dinov2 = DINOv2FeatureExport(load_dinov2(device)).to(device).eval()
        dino_dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
        dino_path = output_dir / "dinov2_feature.onnx"
        export_dino(
            dinov2,
            dino_dummy,
            dino_path,
            args.opset,
            args.exporter,
        )
        if not args.skip_verify:
            log("[verify] Moving DINOv2 verification to CPU...")
            verify_dino(dinov2.cpu(), dino_dummy.cpu(), dino_path)
        del dinov2, dino_dummy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    write_engine_config(output_dir)
    log("Generic ONNX export command finished.")


if __name__ == "__main__":
    main()
