from __future__ import annotations

import json
import pickle
from pathlib import Path


ADAPTER_CONFIG_FILENAME = "patchcore_adapter_config.json"
EXPECTED = [
    (Path("products/screw/models/patchcore_320_l23"), 366, 320, ["layer2", "layer3"]),
    (Path("products/screw/models/patchcore_320_direct_l23"), 320, 320, ["layer2", "layer3"]),
    (Path("products/screw/models/patchcore_320_roi_l23"), 320, 320, ["layer2", "layer3"]),
]


def main():
    print("========== 修复 PatchCore 预处理 metadata ==========")
    repaired = 0

    for model_dir, resize, imagesize, layers in EXPECTED:
        params_file = model_dir / "patchcore_params.pkl"
        index_file = model_dir / "nnscorer_search_index.faiss"

        if not params_file.exists() or not index_file.exists():
            print(f"[跳过] 模型不完整: {model_dir}")
            continue

        with open(params_file, "rb") as f:
            saved = pickle.load(f)

        input_shape = saved.get("input_shape")
        saved_layers = list(saved.get("layers_to_extract_from", []))
        if input_shape is None or int(input_shape[-1]) != imagesize:
            raise RuntimeError(
                f"模型尺寸与预期不一致: {model_dir}, input_shape={input_shape}, "
                f"expected imagesize={imagesize}"
            )
        if saved_layers != layers:
            raise RuntimeError(
                f"模型特征层与预期不一致: {model_dir}, layers={saved_layers}, "
                f"expected={layers}"
            )

        metadata = {
            "format_version": 1,
            "resize": resize,
            "imagesize": imagesize,
            "layers": layers,
        }
        metadata_file = model_dir / ADAPTER_CONFIG_FILENAME
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        repaired += 1
        print(
            f"[OK] {model_dir}: resize={resize}, imagesize={imagesize}, "
            f"layers={'+'.join(layers)}"
        )

    print("==================================================")
    print(f"已写入 metadata: {repaired}/{len(EXPECTED)} 个模型")
    print("现在重新运行: python compare_roi_preprocessing_320.py")
    print("无需重新训练已有 PatchCore Memory Bank。")


if __name__ == "__main__":
    main()
