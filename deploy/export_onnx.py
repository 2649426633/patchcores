from __future__ import annotations

import argparse
import json
import math
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


class PatchCoreFeatureExport(nn.Module):
    """Export the exact PatchCore patch embedding path used by this repository.

    Input is already-resized/normalized NCHW RGB. The output is the final
    PatchCore embedding before nearest-neighbour scoring:

        [B, 3, 320, 320] -> [B, 1600, 1024]

    The graph mirrors patchcore.PatchCore._embed for layer2 + layer3,
    patchsize=3, stride=1, pretrain_dim=1024 and target_dim=1024.
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
            raise ValueError("PatchCore ONNX input_size must be divisible by 16")
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
        # F.unfold: [B, C*P*P, N] -> [B, N, C*P*P]
        return F.unfold(
            x,
            kernel_size=self.patch_size,
            stride=1,
            padding=self.padding,
        ).transpose(1, 2)

    def _mean_map(self, x: torch.Tensor) -> torch.Tensor:
        # Equivalent to patchcore.common.MeanMapper.
        batch, patches, dim = x.shape
        x = x.reshape(batch * patches, 1, dim)
        x = F.adaptive_avg_pool1d(x, self.pretrain_dim).squeeze(1)
        return x.reshape(batch, patches, self.pretrain_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        layer2, layer3 = self._backbone_features(images)

        patch2 = self._patchify(layer2)
        patch3 = self._patchify(layer3)

        # Match PatchCore's spatial interpolation of layer3 patch vectors onto
        # the layer2 patch grid. Every C/P/P element is interpolated independently.
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

        # Equivalent to patchcore.common.Aggregator over two 1024-d features.
        merged = torch.stack((mapped2, mapped3), dim=2)
        b, n, layers, dim = merged.shape
        merged = merged.reshape(b * n, 1, layers * dim)
        merged = F.adaptive_avg_pool1d(merged, self.target_dim).squeeze(1)
        return merged.reshape(b, n, self.target_dim)


class DINOv2FeatureExport(nn.Module):
    """Frozen DINOv2 ViT-S/14 feature exporter used by the defect bank.

    Outputs the two production features currently used by the project:
    L2-normalized CLS and L2-normalized center-patch pooled embeddings.
    """

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

    backbone = models.wide_resnet50_2(weights=None)
    state = torch.load(weight_path, map_location="cpu")
    if isinstance(state, dict) and isinstance(state.get("state_dict"), dict):
        state = state["state_dict"]
    backbone.load_state_dict(state, strict=True)
    backbone.eval().to(device)
    for p in backbone.parameters():
        p.requires_grad_(False)
    return backbone


def load_dinov2(device: torch.device) -> nn.Module:
    repo_dir = REPO_ROOT / "third_party" / "dinov2"
    weights_path = REPO_ROOT / "weights" / "dinov2_vits14_pretrain.pth"
    if not (repo_dir / "hubconf.py").exists():
        raise FileNotFoundError(f"Missing local DINOv2 repository: {repo_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing DINOv2 weights: {weights_path}")

    model = torch.hub.load(
        str(repo_dir),
        "dinov2_vits14",
        source="local",
        pretrained=True,
        weights=str(weights_path),
    )
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_model(
    model: nn.Module,
    dummy: torch.Tensor,
    output_path: Path,
    output_names: list[str],
    opset: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()

    common = dict(
        input_names=["images"],
        output_names=output_names,
        opset_version=opset,
        export_params=True,
    )

    try:
        torch.onnx.export(
            model,
            (dummy,),
            str(output_path),
            dynamo=True,
            optimize=True,
            **common,
        )
        exporter = "dynamo"
    except Exception as exc:
        print(f"[ONNX] dynamo exporter failed: {type(exc).__name__}: {exc}")
        print("[ONNX] retrying with legacy exporter for compatibility...")
        torch.onnx.export(
            model,
            dummy,
            str(output_path),
            dynamo=False,
            do_constant_folding=True,
            **common,
        )
        exporter = "legacy"

    print(f"[ONNX] exported ({exporter}): {output_path.resolve()}")


def verify_onnx(
    model: nn.Module,
    dummy: torch.Tensor,
    onnx_path: Path,
) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError:
        print("[ONNX] verification skipped: install deploy/requirements.txt")
        return

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    with torch.inference_mode():
        torch_outputs = model(dummy)
    if isinstance(torch_outputs, torch.Tensor):
        torch_outputs = (torch_outputs,)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, {"images": dummy.detach().cpu().numpy()})
    if len(torch_outputs) != len(ort_outputs):
        raise RuntimeError("ONNX output count differs from PyTorch")

    for index, (pt, ort_out) in enumerate(zip(torch_outputs, ort_outputs)):
        pt_np = pt.detach().cpu().numpy()
        if pt_np.shape != ort_out.shape:
            raise RuntimeError(
                f"Output {index} shape mismatch: PyTorch={pt_np.shape}, ONNX={ort_out.shape}"
            )
        max_abs = float(np.max(np.abs(pt_np - ort_out)))
        print(f"[ONNX] verify output[{index}] shape={pt_np.shape} max_abs={max_abs:.6g}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export generic PatchCore and DINOv2 feature extractors to ONNX."
    )
    p.add_argument(
        "--output-dir",
        default=str(DEPLOY_ROOT / "models"),
        help="Output folder for generic ONNX engine files.",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--skip-verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== Generic ONNX Export ==========")
    print(f"repo root:   {REPO_ROOT}")
    print(f"output dir:  {output_dir}")
    print(f"device:      {device}")
    print(f"opset:       {args.opset}")
    print("=========================================")

    patchcore = PatchCoreFeatureExport(load_patchcore_backbone(device)).to(device).eval()
    patch_dummy = torch.randn(1, 3, 320, 320, device=device, dtype=torch.float32)
    patch_path = output_dir / "patchcore_feature.onnx"
    export_model(patchcore, patch_dummy, patch_path, ["patch_embeddings"], args.opset)
    if not args.skip_verify:
        verify_onnx(patchcore.cpu(), patch_dummy.cpu(), patch_path)

    dinov2 = DINOv2FeatureExport(load_dinov2(device)).to(device).eval()
    dino_dummy = torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)
    dino_path = output_dir / "dinov2_feature.onnx"
    export_model(
        dinov2,
        dino_dummy,
        dino_path,
        ["cls_embedding", "center_embedding"],
        args.opset,
    )
    if not args.skip_verify:
        verify_onnx(dinov2.cpu(), dino_dummy.cpu(), dino_path)

    config = {
        "format_version": 1,
        "patchcore": {
            "file": patch_path.name,
            "input": "images",
            "input_shape": [1, 3, 320, 320],
            "output": "patch_embeddings",
            "output_shape": [1, 1600, 1024],
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
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ONNX] engine config: {config_path.resolve()}")
    print("\nGeneric ONNX engine export finished.")


if __name__ == "__main__":
    main()
