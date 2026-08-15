from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSPACEAD_SRC = REPO_ROOT / "third_party" / "subspacead" / "src"
if str(SUBSPACEAD_SRC) not in sys.path:
    sys.path.insert(0, str(SUBSPACEAD_SRC))

from subspacead.core.extractor import FeatureExtractor
from subspacead.core.pca import PCAModel
from subspacead.data.transforms import get_augmentation_transform
from subspacead.post_process.scoring import calculate_anomaly_scores, post_process_map
from subspacead.utils.common import min_max_norm, topk_mean


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Official paper / benchmark configuration.  These are intentionally fixed for
# the first smoke test; this script is not an ablation runner.
IMAGE_RES = 672
LAYERS = [-12, -13, -14, -15, -16, -17, -18]
AGG_METHOD = "mean"
PCA_EXPLAINED_VARIANCE = 0.99
AUGMENTATION = ["rotate"]
DEFAULT_AUG_COUNT = 30
IMAGE_SCORE_TOP_FRACTION = 0.01


def log(message: str) -> None:
    print(message, flush=True)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(folder)
    images = sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise RuntimeError(f"No images found in: {folder}")
    return images


def save_pca_model(output_dir: Path, pca_params: dict, selected_train: list[Path], model_ckpt: str, seed: int, aug_count: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "pca_model.npz",
        mu=np.asarray(pca_params["mu"], dtype=np.float64),
        components=np.asarray(pca_params["components"], dtype=np.float64),
        eigvals=np.asarray(pca_params["eigvals"], dtype=np.float64),
        sqrt_eig=np.asarray(pca_params["sqrt_eig"], dtype=np.float64),
        k=np.asarray([int(pca_params["k"])], dtype=np.int64),
        eps=np.asarray([float(pca_params["eps"])], dtype=np.float64),
    )
    config = {
        "method": "SubspaceAD",
        "source": "CLendering/SubspaceAD official core, invoked without modifying third_party/subspacead",
        "model_ckpt": model_ckpt,
        "image_res": IMAGE_RES,
        "layers": LAYERS,
        "aggregation": AGG_METHOD,
        "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
        "augmentation": AUGMENTATION,
        "augmentation_count": aug_count,
        "image_score_top_fraction": IMAGE_SCORE_TOP_FRACTION,
        "seed": seed,
        "selected_normal_images": [str(p) for p in selected_train],
        "pca_components": int(pca_params["k"]),
    }
    (output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def save_visuals(test_image: Path, anomaly_map: np.ndarray, output_dir: Path) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = output_dir / f"{test_image.stem}_anomaly_raw.npy"
    np.save(raw_path, anomaly_map.astype(np.float32))

    normalized = np.asarray(min_max_norm(anomaly_map), dtype=np.float32)
    map_u8 = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(map_u8, cv2.COLORMAP_JET)

    heat_path = output_dir / f"{test_image.stem}_heatmap.jpg"
    cv2.imwrite(str(heat_path), heat)

    original = cv2.imread(str(test_image), cv2.IMREAD_COLOR)
    if original is None:
        raise RuntimeError(f"OpenCV failed to read: {test_image}")
    heat_original = cv2.resize(
        heat, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    overlay = cv2.addWeighted(original, 0.55, heat_original, 0.45, 0.0)
    overlay_path = output_dir / f"{test_image.stem}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    return raw_path, heat_path, overlay_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal SubspaceAD smoke test using the official copied core. "
            "Fits PCA on a few normal images and evaluates one test image."
        )
    )
    parser.add_argument("--normal-dir", required=True, help="Folder containing only normal images")
    parser.add_argument("--image", required=True, help="One image to inspect")
    parser.add_argument("--shots", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument(
        "--model-ckpt",
        default="facebook/dinov2-with-registers-giant",
        help="HuggingFace model id or a local HuggingFace model directory",
    )
    parser.add_argument("--aug-count", type=int, default=DEFAULT_AUG_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "industrial_anomaly" / "outputs" / "subspacead_smoke"),
    )
    args = parser.parse_args()

    if args.aug_count < 0:
        raise ValueError("--aug-count must be >= 0")

    normal_dir = Path(args.normal_dir).expanduser().resolve()
    test_image = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if not test_image.exists():
        raise FileNotFoundError(test_image)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    normal_images = list_images(normal_dir)
    if len(normal_images) < args.shots:
        raise RuntimeError(
            f"Requested {args.shots}-shot but only {len(normal_images)} normal images exist"
        )

    shuffled = normal_images.copy()
    random.shuffle(shuffled)
    train_paths = shuffled[: args.shots]

    log("========== SubspaceAD Smoke Test ==========")
    log(f"normal dir:   {normal_dir}")
    log(f"test image:   {test_image}")
    log(f"shots:        {args.shots}")
    log(f"augmentations:{args.aug_count} rotation views / normal image")
    log(f"model:        {args.model_ckpt}")
    log(f"resolution:   {IMAGE_RES}")
    log(f"layers:       {LAYERS}")
    log(f"aggregation:  {AGG_METHOD}")
    log(f"PCA EV:       {PCA_EXPLAINED_VARIANCE}")
    log(f"device:       {'cuda' if torch.cuda.is_available() else 'cpu'}")
    log("===========================================")
    for index, path in enumerate(train_paths, start=1):
        log(f"normal[{index}]={path.name}")

    log("[1/4] Loading official SubspaceAD FeatureExtractor...")
    extractor = FeatureExtractor(args.model_ckpt)

    log("[2/4] Reading feature shape...")
    first = Image.open(train_paths[0]).convert("RGB")
    temp_tokens, (h_p, w_p), _ = extractor.extract_tokens(
        [first],
        IMAGE_RES,
        LAYERS,
        AGG_METHOD,
        [],
        False,
        use_clahe=False,
        dino_saliency_layer=0,
    )
    feature_dim = int(temp_tokens.shape[-1])
    tokens_per_image = int(h_p * w_p)
    multiplier = 1 + args.aug_count
    total_train_images = len(train_paths) * multiplier
    total_tokens = total_train_images * tokens_per_image
    num_batches = total_train_images
    log(
        f"[2/4] feature_dim={feature_dim}, grid={h_p}x{w_p}, "
        f"tokens/image={tokens_per_image}, PCA tokens={total_tokens}"
    )

    aug_transform = (
        get_augmentation_transform(AUGMENTATION, IMAGE_RES)
        if args.aug_count > 0
        else None
    )

    # This intentionally follows the official full-image few-shot generator:
    # original image + N random rotation views. PCAModel calls the generator
    # twice (mean pass, covariance pass), as in the official benchmark code.
    def feature_generator():
        for path in train_paths:
            pil_img = Image.open(path).convert("RGB")
            images_to_process = [pil_img]
            if aug_transform is not None:
                for _ in range(args.aug_count):
                    images_to_process.append(aug_transform(pil_img))

            for image in images_to_process:
                tokens, _, _ = extractor.extract_tokens(
                    [image],
                    IMAGE_RES,
                    LAYERS,
                    AGG_METHOD,
                    [],
                    False,
                    use_clahe=False,
                    dino_saliency_layer=0,
                )
                yield tokens.reshape(-1, feature_dim)

    log("[3/4] Fitting official PCA model...")
    pca_model = PCAModel(k=None, ev=PCA_EXPLAINED_VARIANCE, whiten=False)
    pca_params = pca_model.fit(
        feature_generator,
        feature_dim,
        total_tokens,
        num_batches,
    )
    log(f"[3/4] PCA components selected: k={pca_params['k']}")
    save_pca_model(output_dir, pca_params, train_paths, args.model_ckpt, args.seed, args.aug_count)

    log("[4/4] Inspecting test image...")
    test_pil = Image.open(test_image).convert("RGB")
    test_tokens, (test_h, test_w), _ = extractor.extract_tokens(
        [test_pil],
        IMAGE_RES,
        LAYERS,
        AGG_METHOD,
        [],
        False,
        use_clahe=False,
        dino_saliency_layer=0,
    )
    flattened = test_tokens.reshape(test_h * test_w, feature_dim)
    patch_scores = calculate_anomaly_scores(
        flattened,
        pca_params,
        method="reconstruction",
        drop_k=0,
    )
    anomaly_map_low = patch_scores.reshape(test_h, test_w)
    anomaly_map = post_process_map(anomaly_map_low, IMAGE_RES)
    image_score = topk_mean(anomaly_map, frac=IMAGE_SCORE_TOP_FRACTION)

    raw_path, heat_path, overlay_path = save_visuals(test_image, anomaly_map, output_dir)

    result = {
        "image": str(test_image),
        "image_score_tvar_top_1pct": float(image_score),
        "patch_score_min": float(np.min(patch_scores)),
        "patch_score_max": float(np.max(patch_scores)),
        "pca_components": int(pca_params["k"]),
        "feature_dim": feature_dim,
        "patch_grid": [test_h, test_w],
        "raw_map": str(raw_path),
        "heatmap": str(heat_path),
        "overlay": str(overlay_path),
    }
    (output_dir / f"{test_image.stem}_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    log("")
    log("========== RESULT ==========")
    log(f"image score (top 1%): {image_score:.8f}")
    log(f"patch score min/max:  {np.min(patch_scores):.8f} / {np.max(patch_scores):.8f}")
    log(f"PCA components:       {pca_params['k']}")
    log(f"heatmap:              {heat_path}")
    log(f"overlay:              {overlay_path}")
    log(f"model:                {output_dir / 'pca_model.npz'}")
    log("============================")


if __name__ == "__main__":
    main()
