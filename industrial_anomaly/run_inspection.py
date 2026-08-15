from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.anomaly.tiled import (
    crop_square_with_margin,
    inspect_tiled_patchcore,
    load_inspection_config,
    save_tiled_heatmap_overlay,
)
from app.defect.defect_bank import DefectExemplarBank
from app.defect.patchcore_dinov2_pipeline import PatchCoreDINOv2Pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = HERE / path
    return path.resolve()


def collect_images(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in IMAGE_EXTS:
            raise ValueError(f"Unsupported image type: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )
    if not images:
        raise RuntimeError(f"No supported images found in: {path}")
    return images


def rank_scores(scores: dict[str, float]) -> dict:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top1_class, top1 = ranked[0]
    if len(ranked) > 1:
        top2_class, top2 = ranked[1]
        margin = float(top1 - top2)
    else:
        top2_class, top2, margin = None, float("-inf"), float("inf")
    return {
        "predicted_class": top1_class,
        "top1_similarity": float(top1),
        "top2_class": top2_class,
        "top2_similarity": float(top2),
        "margin": margin,
    }


def classify_roi(
    pipeline: PatchCoreDINOv2Pipeline,
    cls_bank: DefectExemplarBank,
    center_bank: DefectExemplarBank,
    roi: Image.Image,
    center_fraction: float,
) -> dict:
    cls_embedding = pipeline.embed_roi(roi, feature_mode="cls")
    center_embedding = pipeline.embed_roi(
        roi,
        feature_mode="patch_center",
        center_fraction=center_fraction,
    )
    cls_result = cls_bank.predict_embedding(cls_embedding)
    center_result = center_bank.predict_embedding(center_embedding)
    fused_scores = {
        name: 0.50 * cls_result["class_scores"][name]
        + 0.50 * center_result["class_scores"][name]
        for name in cls_bank.classes
    }
    result = rank_scores(fused_scores)
    result["class_scores"] = fused_scores
    return result


def save_marked(
    image_path: Path,
    bbox: tuple[int, int, int, int] | None,
    label: str,
    score: float,
    similarity: float | None,
    output_path: Path,
) -> None:
    rgb = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = canvas.shape[:2]
    thickness = max(2, int(round(min(w, h) / 350.0)))
    font_scale = max(0.55, min(1.3, min(w, h) / 1500.0))

    text = f"{label} | PatchCore={score:.3f}"
    if similarity is not None:
        text += f" | sim={similarity:.3f}"

    if bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), thickness)
        origin = (max(0, x1), max(30, y1 - 10))
    else:
        origin = (20, 50)

    cv2.putText(
        canvas,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(output_path.suffix or ".jpg", canvas)
    if not ok:
        raise RuntimeError(f"Failed to encode marked image: {output_path}")
    encoded.tofile(str(output_path))


def inspect_one(
    image_path: Path,
    *,
    pipeline: PatchCoreDINOv2Pipeline,
    cls_bank: DefectExemplarBank,
    center_bank: DefectExemplarBank,
    inspection_mode: str,
    tile_fraction: float,
    tile_overlap: float,
    bbox_relative_threshold: float,
    roi_margin: float,
    center_fraction: float,
    anomaly_threshold: float | None,
    output_dir: Path,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    if inspection_mode == "tiled":
        tiled = inspect_tiled_patchcore(
            pipeline.patchcore,
            image_path,
            tile_fraction=tile_fraction,
            overlap=tile_overlap,
            relative_threshold=bbox_relative_threshold,
        )
        anomaly_score = float(tiled["anomaly_score"])
        primary_bbox = tuple(tiled["regions"][0]["bbox"]) if tiled["regions"] else None
        roi = (
            crop_square_with_margin(
                Image.open(image_path).convert("RGB"),
                primary_bbox,
                margin=roi_margin,
            )
            if primary_bbox is not None
            else None
        )
        save_tiled_heatmap_overlay(
            image_path,
            tiled["tile_results"],
            output_dir / "heatmap.jpg",
        )
    else:
        legacy = pipeline.extract_roi(image_path)
        anomaly_score = float(legacy["anomaly_score"])
        primary_bbox = tuple(int(v) for v in legacy["original_bbox"])
        roi = legacy["roi"]

    if anomaly_threshold is not None and anomaly_score < anomaly_threshold:
        decision = "PASS"
        predicted = None
        similarity = None
        margin = None
        final = "PASS"
    elif roi is None:
        decision = "NG" if anomaly_threshold is not None else "UNCALIBRATED"
        predicted = None
        similarity = None
        margin = None
        final = "NO_LOCALIZED_REGION"
    else:
        defect = classify_roi(
            pipeline,
            cls_bank,
            center_bank,
            roi,
            center_fraction,
        )
        predicted = defect["predicted_class"]
        similarity = float(defect["top1_similarity"])
        margin = float(defect["margin"])
        if anomaly_threshold is None:
            decision = "UNCALIBRATED"
            final = f"KNOWN_DEFECT_CANDIDATE: {predicted}"
        else:
            decision = "NG"
            final = f"NG: {predicted}"
        roi.save(output_dir / "roi.png")

    marked_path = output_dir / "marked.jpg"
    save_marked(
        image_path,
        primary_bbox,
        final,
        anomaly_score,
        similarity,
        marked_path,
    )

    payload = {
        "image": str(image_path),
        "inspection_mode": inspection_mode,
        "patchcore_anomaly_score": anomaly_score,
        "anomaly_threshold": anomaly_threshold,
        "anomaly_decision": decision,
        "bbox_original_image": list(primary_bbox) if primary_bbox is not None else None,
        "predicted_known_defect": predicted,
        "top1_similarity": similarity,
        "margin": margin,
        "final_result": final,
        "marked_image": str(marked_path.resolve()),
    }
    (output_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified inspection for one image or a folder."
    )
    p.add_argument("input", help="Image file or folder")
    p.add_argument("--product", required=True)
    p.add_argument("--patchcore-model-dir", default=None)
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--output-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--bbox-relative-threshold", type=float, default=0.78)
    p.add_argument("--roi-margin", type=float, default=0.50)
    p.add_argument("--center-fraction", type=float, default=0.50)
    p.add_argument("--anomaly-threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = resolve_path(args.input)
    images = collect_images(input_path)

    product_dir = HERE / "products" / args.product
    patchcore_model_dir = (
        resolve_path(args.patchcore_model_dir)
        if args.patchcore_model_dir
        else product_dir / "models" / "patchcore"
    )
    bank_dir = (
        resolve_path(args.bank_dir)
        if args.bank_dir
        else product_dir / "models" / "defect_bank"
    )
    output_root = (
        resolve_path(args.output_dir)
        if args.output_dir
        else HERE / "outputs" / args.product / input_path.stem
    )

    if not patchcore_model_dir.exists():
        raise FileNotFoundError(f"PatchCore model not found: {patchcore_model_dir}")
    if not (bank_dir / "cls").exists() or not (bank_dir / "center").exists():
        raise FileNotFoundError(f"Defect bank not found: {bank_dir}")

    inspection_cfg = load_inspection_config(patchcore_model_dir)
    inspection_mode = str(inspection_cfg.get("mode", "center_crop"))
    tile_fraction = float(inspection_cfg.get("tile_fraction", 0.75))
    tile_overlap = float(inspection_cfg.get("tile_overlap", 0.25))

    pipeline = PatchCoreDINOv2Pipeline(
        patchcore_model_dir=patchcore_model_dir,
        bank_dir=None,
        device=args.device,
        bbox_relative_threshold=args.bbox_relative_threshold,
        roi_margin=args.roi_margin,
        center_fraction=args.center_fraction,
    )
    pipeline.load()

    cls_bank = DefectExemplarBank.load(bank_dir / "cls")
    center_bank = DefectExemplarBank.load(bank_dir / "center")
    if cls_bank.classes != center_bank.classes:
        raise RuntimeError("CLS and Center banks contain different classes")

    rows: list[dict] = []
    for index, image_path in enumerate(images, start=1):
        image_output = output_root if len(images) == 1 else output_root / image_path.stem
        row = inspect_one(
            image_path,
            pipeline=pipeline,
            cls_bank=cls_bank,
            center_bank=center_bank,
            inspection_mode=inspection_mode,
            tile_fraction=tile_fraction,
            tile_overlap=tile_overlap,
            bbox_relative_threshold=args.bbox_relative_threshold,
            roi_margin=args.roi_margin,
            center_fraction=args.center_fraction,
            anomaly_threshold=args.anomaly_threshold,
            output_dir=image_output,
        )
        rows.append(row)
        print(
            f"[{index}/{len(images)}] {image_path.name} "
            f"PatchCore={row['patchcore_anomaly_score']:.4f} "
            f"class={row['predicted_known_defect'] or '-'} "
            f"sim={row['top1_similarity'] if row['top1_similarity'] is not None else '-'}"
        )

    if len(rows) > 1:
        output_root.mkdir(parents=True, exist_ok=True)
        csv_path = output_root / "results.csv"
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"CSV: {csv_path.resolve()}")

    print(f"Output: {output_root.resolve()}")


if __name__ == "__main__":
    main()
