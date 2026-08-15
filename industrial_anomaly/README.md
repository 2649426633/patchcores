# Python training / validation workflow

此目录只负责训练电脑上的 Python 流程。

现在只保留三个主入口：

```text
industrial_anomaly/
├── train_patchcore.py       # 正常图 → PatchCore + Greedy Coreset + FAISS
├── build_defect_bank.py     # 已知缺陷 → PatchCore ROI → DINOv2 exemplar bank
├── run_inspection.py        # 单图 / 文件夹统一验收
└── README.md
```

> 不使用 `inspect.py` 作为脚本名，因为它会覆盖 Python 标准库 `inspect`，导致 NumPy / PyTorch 等第三方库导入异常。

## 1. 数据结构

数据放仓库根目录 `data/`：

```text
data/<product>/
├── good/          # 仅正常图
├── defect_a/      # 已知缺陷类别
├── defect_b/
└── test/          # 独立测试图
```

例如：

```text
data/phone/
├── good/
├── shao1/
├── shao2/
├── shao3/
└── test/
```

## 2. 训练 PatchCore

高分辨率工业图推荐 `tiled`：

```powershell
cd industrial_anomaly

python train_patchcore.py ^
  --product phone ^
  --normal-dir ..\data\phone\good ^
  --mode tiled
```

默认核心参数：

```text
imagesize      = 320
tile_fraction  = 0.75
tile_overlap   = 0.25
layers         = layer2 + layer3
coreset        = 0.10
```

训练流程：

```text
NORMAL images only
→ WideResNet50-2
→ PatchCore patch embeddings
→ ApproximateGreedyCoresetSampler
→ FAISS IndexFlatL2
```

输出：

```text
products/<product>/models/patchcore/
├── nnscorer_search_index.faiss
├── patchcore_params.pkl
├── patchcore_adapter_config.json
└── inspection_config.json
```

`nnscorer_search_index.faiss` 是部署时必须保留的原始正常 Memory 来源。

## 3. 建立已知缺陷 bank

```powershell
python build_defect_bank.py ^
  --product phone ^
  --defects-dir ..\data\phone ^
  --classes shao1 shao2 shao3 ^
  --shots 10
```

流程：

```text
known defect image
→ 已训练 PatchCore 定位异常
→ 裁 ROI
→ frozen DINOv2
  ├─ CLS
  └─ Patch Center
→ exemplar banks
```

输出：

```text
products/<product>/models/defect_bank/
├── cls/
│   ├── embeddings.npz
│   └── metadata.json
├── center/
│   ├── embeddings.npz
│   └── metadata.json
├── bank_config.json
└── support_rois/
```

这些 `.npz` 是部署时 `defect_cls.bin` / `defect_center.bin` 的唯一正式来源。

## 4. Python 验收

现在单图和文件夹统一用 `run_inspection.py`。

文件夹：

```powershell
python run_inspection.py ..\data\phone\test --product phone
```

单图：

```powershell
python run_inspection.py ..\data\phone\test\Image_3.bmp --product phone
```

脚本会自动读取：

```text
products/<product>/models/patchcore/inspection_config.json
```

如果模型是 `tiled`，就自动执行完整 tiled 检测，不再走旧 center-crop 单图分支。

当前推荐定位参数：

```text
bbox_relative_threshold = 0.78
tile_fraction           = 0.75
tile_overlap            = 0.25
roi_margin              = 0.50
center_fraction         = 0.50
```

输出每张图片自己的目录：

```text
outputs/<product>/<input_name>/
└── <image>/
    ├── marked.jpg
    ├── roi.png
    ├── heatmap.jpg       # tiled 时
    └── result.json
```

文件夹检测还会生成 `results.csv`。

## 5. 调试顺序

结果不对时严格按顺序检查：

```text
PatchCore heatmap
    ↓
BBox
    ↓
ROI
    ↓
DINO CLS / Center
    ↓
Similarity / Class
```

判断：

- 热图不在真实缺陷：先检查 PatchCore normal 数据、tile 尺度、特征。
- 热图正确但框不对：检查后处理 / threshold / tile merge。
- 框正确但类别错：再检查 DINO bank 与 ROI。

不要用分类器问题掩盖前面的定位问题。

## 6. PASS / NG

`--bbox-relative-threshold` 只是**定位阈值**，不是 PASS/NG 阈值。

生产 PASS/NG 阈值必须从独立校准数据获得，不能用最终 `test/` 标签反推。

## 7. 部署到 C# / WinForms

Python 效果确认后，不要让 C# 重新训练。

回仓库根目录运行：

```powershell
python deploy\convert_python_product.py ^
  --product phone ^
  --patchcore-model-dir industrial_anomaly\products\phone\models\patchcore ^
  --defect-bank-dir industrial_anomaly\products\phone\models\defect_bank ^
  --output-dir deploy\products\phone ^
  --bbox-relative-threshold 0.78 ^
  --roi-margin 0.50 ^
  --copy-support-rois
```

部署端只使用转换后的：

```text
patchcore_memory.bin
defect_cls.bin
defect_center.bin
product_model.json
```

详细说明见 `../deploy/README.md`。
