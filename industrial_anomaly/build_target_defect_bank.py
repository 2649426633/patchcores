from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.defect.defect_bank import DefectExemplarBank
from app.defect.dinov2_adapter import DINOv2Adapter


def resolve_path(value: str | Path) -> Path:
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = REPO_ROOT / p
    return p.resolve()


def crop_with_margin(image: Image.Image, box: list[int], margin: float) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(bw, bh) * (1.0 + 2.0 * float(margin))
    side = max(8, int(round(side)))
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    fill = tuple(int(v) for v in np.median(np.asarray(image), axis=(0, 1)))
    roi = Image.new("RGB", (side, side), fill)
    sx1, sy1 = max(0, left), max(0, top)
    sx2, sy2 = min(w, right), min(h, bottom)
    if sx2 > sx1 and sy2 > sy1:
        roi.paste(image.crop((sx1, sy1, sx2, sy2)), (sx1 - left, sy1 - top))
    return roi


def main() -> None:
    p = argparse.ArgumentParser(
        description="Build a known target-defect DINOv2 bank from manually annotated GT boxes."
    )
    p.add_argument("--annotations", required=True)
    p.add_argument("--product", required=True)
    p.add_argument("--bank-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--roi-margin", type=float, default=0.35)
    p.add_argument("--center-fraction", type=float, default=0.60)
    args = p.parse_args()

    annotations_path = resolve_path(args.annotations)
    if not annotations_path.exists():
        raise FileNotFoundError(annotations_path)
    doc = json.loads(annotations_path.read_text(encoding="utf-8"))
    records = [v for v in doc.get("images", {}).values() if v.get("boxes")]
    if not records:
        raise RuntimeError("No annotated boxes found")

    bank_dir = (
        resolve_path(args.bank_dir)
        if args.bank_dir
        else HERE / "products" / args.product / "models" / "target_defect_bank"
    )
    bank_dir.mkdir(parents=True, exist_ok=True)

    adapter = DINOv2Adapter(device=args.device)
    adapter.load()

    cls_embeddings: list[np.ndarray] = []
    center_embeddings: list[np.ndarray] = []
    labels: list[str] = []
    sources: list[str] = []
    boxes_meta: list[list[int]] = []
    crop_sizes: dict[str, list[list[int]]] = {}

    print("========== Build TARGET Defect Bank ==========")
    print(f"annotations: {annotations_path}")
    print(f"records:     {len(records)}")
    print(f"bank:        {bank_dir}")
    print("ROI source:  MANUAL GROUND-TRUTH boxes (no PatchCore/SubspaceAD ROI)")
    print("==============================================")

    count = 0
    for rec in records:
        class_name = str(rec["class"])
        image_path = Path(rec["image"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        image = Image.open(image_path).convert("RGB")
        crop_sizes.setdefault(class_name, [])

        for box_index, box in enumerate(rec["boxes"]):
            x1, y1, x2, y2 = [int(v) for v in box]
            crop_sizes[class_name].append([x2 - x1, y2 - y1])
            roi = crop_with_margin(image, [x1, y1, x2, y2], args.roi_margin)

            cls = adapter.embed(roi, feature_mode="cls")
            center = adapter.embed(
                roi,
                feature_mode="patch_center",
                center_fraction=args.center_fraction,
            )
            cls_embeddings.append(cls)
            center_embeddings.append(center)
            labels.append(class_name)
            sources.append(f"{image_path}#box{box_index}")
            boxes_meta.append([x1, y1, x2, y2])

            roi_path = bank_dir / "gt_rois" / class_name / f"{image_path.stem}_box{box_index}.png"
            roi_path.parent.mkdir(parents=True, exist_ok=True)
            roi.save(roi_path)
            count += 1
            print(f"+ {class_name}: {image_path.name} box={box}")

    cls_bank = DefectExemplarBank(np.stack(cls_embeddings), labels, sources)
    center_bank = DefectExemplarBank(np.stack(center_embeddings), labels, sources)
    cls_bank.save(bank_dir / "cls")
    center_bank.save(bank_dir / "center")

    median_sizes = {}
    for class_name, sizes in crop_sizes.items():
        arr = np.asarray(sizes, dtype=np.float32)
        median_sizes[class_name] = [
            int(round(float(np.median(arr[:, 0])))),
            int(round(float(np.median(arr[:, 1])))),
        ]

    config = {
        "format_version": 1,
        "product": args.product,
        "classes": sorted(set(labels)),
        "num_gt_boxes": count,
        "source": "manual_ground_truth_boxes",
        "annotations": str(annotations_path),
        "roi_margin": float(args.roi_margin),
        "center_fraction": float(args.center_fraction),
        "features": ["dinov2_cls", "dinov2_patch_center"],
        "fusion_cls_weight": 0.5,
        "fusion_center_weight": 0.5,
        "median_defect_size_px": median_sizes,
        "boxes": boxes_meta,
    }
    (bank_dir / "bank_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("========== TARGET BANK DONE ==========")
    print(f"classes: {sorted(set(labels))}")
    print(f"boxes:   {count}")
    print(f"saved:   {bank_dir}")


if __name__ == "__main__":
    main()
