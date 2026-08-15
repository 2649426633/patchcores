from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def load_existing(path: Path) -> dict:
    if not path.exists():
        return {"format_version": 1, "images": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    p = argparse.ArgumentParser(
        description="Draw true target-defect boxes for known defect classes."
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--classes", nargs="+", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max-per-class", type=int, default=0, help="0 means all images")
    args = p.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(data_dir)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else data_dir / "target_defect_boxes.json"
    )
    doc = load_existing(output)
    images_doc = doc.setdefault("images", {})

    tasks: list[tuple[str, Path]] = []
    for class_name in args.classes:
        folder = data_dir / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"class folder not found: {folder}")
        items = image_files(folder)
        if args.max_per_class > 0:
            items = items[: args.max_per_class]
        tasks.extend((class_name, path) for path in items)

    print("Controls: drag mouse = add box, U = undo, R = clear, N/Enter = save+next, S = skip, Q/Esc = quit")
    print(f"output: {output}")

    quit_all = False
    for index, (class_name, image_path) in enumerate(tasks, 1):
        key_id = str(image_path)
        existing = images_doc.get(key_id, {})
        boxes = [list(map(int, box)) for box in existing.get("boxes", [])]

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable: {image_path}")
            continue
        h, w = image.shape[:2]

        view_scale = min(1.0, 1600.0 / max(w, h))
        vw, vh = max(1, int(round(w * view_scale))), max(1, int(round(h * view_scale)))

        state = {"dragging": False, "start": None, "current": None}
        window = "Target Defect Annotation"

        def mouse(event, x, y, flags, param):
            ox = int(round(x / view_scale))
            oy = int(round(y / view_scale))
            ox = max(0, min(w - 1, ox))
            oy = max(0, min(h - 1, oy))
            if event == cv2.EVENT_LBUTTONDOWN:
                state["dragging"] = True
                state["start"] = (ox, oy)
                state["current"] = (ox, oy)
            elif event == cv2.EVENT_MOUSEMOVE and state["dragging"]:
                state["current"] = (ox, oy)
            elif event == cv2.EVENT_LBUTTONUP and state["dragging"]:
                state["dragging"] = False
                x1, y1 = state["start"]
                x2, y2 = ox, oy
                x1, x2 = sorted((x1, x2))
                y1, y2 = sorted((y1, y2))
                if x2 - x1 >= 3 and y2 - y1 >= 3:
                    boxes.append([x1, y1, x2, y2])
                state["start"] = None
                state["current"] = None

        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, vw, vh)
        cv2.setMouseCallback(window, mouse)

        while True:
            canvas = image.copy()
            for n, (x1, y1, x2, y2) in enumerate(boxes, 1):
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), max(2, int(min(w, h) / 800)))
                cv2.putText(canvas, str(n), (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            if state["dragging"] and state["start"] and state["current"]:
                x1, y1 = state["start"]
                x2, y2 = state["current"]
                cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 255), max(2, int(min(w, h) / 800)))

            title = f"[{index}/{len(tasks)}] {class_name} | {image_path.name} | boxes={len(boxes)}"
            cv2.putText(canvas, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            display = cv2.resize(canvas, (vw, vh), interpolation=cv2.INTER_AREA) if view_scale < 1.0 else canvas
            cv2.imshow(window, display)
            key = cv2.waitKey(20) & 0xFF

            if key in (ord("u"), ord("U")):
                if boxes:
                    boxes.pop()
            elif key in (ord("r"), ord("R")):
                boxes.clear()
            elif key in (ord("n"), ord("N"), 13):
                images_doc[key_id] = {
                    "class": class_name,
                    "image": key_id,
                    "width": w,
                    "height": h,
                    "boxes": boxes,
                }
                output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
                break
            elif key in (ord("s"), ord("S")):
                break
            elif key in (ord("q"), ord("Q"), 27):
                output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
                quit_all = True
                break

        cv2.destroyWindow(window)
        if quit_all:
            break

    output.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    annotated = sum(1 for item in images_doc.values() if item.get("boxes"))
    box_count = sum(len(item.get("boxes", [])) for item in images_doc.values())
    print(f"saved: {output}")
    print(f"annotated images: {annotated}, boxes: {box_count}")


if __name__ == "__main__":
    main()
