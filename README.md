# Industrial Anomaly Detection — PatchCore + DINOv2 + ONNX/C#

工业异常检测项目，最终架构分成两个阶段：

1. **训练电脑（Python）**：PatchCore 只用正常图做无监督训练；DINOv2 使用 PatchCore 定位后的已知缺陷 ROI 建 exemplar bank。
2. **工业电脑（C# / WinForms）**：只运行 ONNX + 从原 Python 模型无损转换出来的产品 `.bin`，不在现场重新训练或随机重建 Memory Bank。

> 当前正式原则：**保留原 Python 的 ApproximateGreedyCoreset + FAISS Memory 和原 DINO defect bank。废弃 C# `bounded_reservoir` 产品建模方案。**

---

## 1. Repository layout

```text
patchcores/
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/                         # 本地数据，不上传图片
│   └── README.md
│
├── weights/                      # 本地模型权重，不上传 .pth
│   └── README.md
│
├── third_party/                  # 本地第三方源码
│   └── README.md                 # DINOv2 应放 third_party/dinov2/
│
├── patchcore/                    # PatchCore 核心实现；不要随意修改
│
├── app/                          # Python 适配层
│   ├── anomaly/                  # PatchCore adapter / tiled / 后处理
│   └── defect/                   # DINOv2 adapter / defect bank / pipeline
│
├── industrial_anomaly/           # Python 训练与验收入口
│   ├── train_patchcore.py
│   ├── build_defect_bank.py
│   ├── inspect.py
│   └── README.md
│
├── deploy/                       # ONNX/C# 部署
│   ├── export_onnx.py
│   ├── convert_python_product.py
│   ├── requirements.txt
│   ├── README.md
│   ├── models/                   # 本地 ONNX + engine_config.json
│   │   └── README.md
│   ├── products/                 # 原 Python 产品模型转换后的部署包
│   │   └── README.md
│   └── csharp/
│       ├── IndustrialAnomaly.Runtime/
│       └── IndustrialAnomaly.Console/
│
└── third_party_licenses/         # 第三方许可证
```

---

## 2. 本地缺失内容放哪里

仓库不会提交大权重、数据、ONNX 和产品模型。clone 后按下面放置。

### 2.1 Python 权重

```text
weights/
├── wide_resnet50_2-95faca4d.pth
└── dinov2_vits14_pretrain.pth
```

### 2.2 DINOv2 官方源码

```text
third_party/
└── dinov2/
    ├── hubconf.py
    └── dinov2/
```

### 2.3 数据集

```text
data/
└── <product>/
    ├── good/          # 正常图，只用于 PatchCore
    ├── <defect_a>/    # 已知缺陷类别
    ├── <defect_b>/
    └── test/          # 独立测试集
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

### 2.4 ONNX 引擎

导出后应为：

```text
deploy/models/
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json
```

### 2.5 WinForms 产品模型包

必须由原 Python 模型转换得到：

```text
deploy/products/<product>/
├── patchcore_memory.bin
├── defect_cls.bin
├── defect_center.bin
├── product_model.json
├── conversion_report.json
└── support_rois/          # 可选
```

正式产品必须标记：

```text
ProductModelSource = python_export
PatchCoreMemoryStrategy = python_faiss_memory_exact
```

---

## 3. Python 环境

Python 3.10 推荐。

先按你的 CUDA 环境单独安装匹配版本的 `torch` / `torchvision`，然后：

```powershell
pip install -r requirements.txt
```

ONNX 导出额外依赖：

```powershell
pip install -r deploy\requirements.txt
```

---

## 4. 训练一个新产品

下面以 `bottle` 为例。

数据：

```text
data/bottle/
├── good/
├── scratch/
├── crack/
├── dirty/
└── test/
```

### 4.1 PatchCore：只用正常图

```powershell
cd industrial_anomaly

python train_patchcore.py ^
  --product bottle ^
  --normal-dir ..\data\bottle\good ^
  --mode tiled
```

核心流程：

```text
normal images
→ WideResNet50-2
→ patch embeddings
→ ApproximateGreedyCoresetSampler
→ FAISS IndexFlatL2 Memory Bank
```

生成：

```text
industrial_anomaly/products/bottle/models/patchcore/
├── nnscorer_search_index.faiss
├── patchcore_params.pkl
├── patchcore_adapter_config.json
└── inspection_config.json
```

### 4.2 建立已知缺陷 DINOv2 bank

```powershell
python build_defect_bank.py ^
  --product bottle ^
  --defects-dir ..\data\bottle ^
  --classes scratch crack dirty ^
  --shots 10
```

生成：

```text
industrial_anomaly/products/bottle/models/defect_bank/
├── cls/
│   ├── embeddings.npz
│   └── metadata.json
├── center/
│   ├── embeddings.npz
│   └── metadata.json
├── bank_config.json
└── support_rois/
```

DINOv2 本身冻结，不训练分类头。类别分数为 exemplar cosine similarity，当前使用：

```text
0.50 × CLS + 0.50 × Patch Center
```

---

## 5. 先在 Python 验收模型

单张和文件夹统一使用一个入口：

```powershell
python inspect.py ..\data\bottle\test --product bottle
```

也可以检测一张：

```powershell
python inspect.py ..\data\bottle\test\Image_1.bmp --product bottle
```

`tiled` 模型会自动读取 `inspection_config.json` 并覆盖整张高分辨率图片。

当前定位默认：

```text
BBoxRelativeThreshold = 0.78
TileFraction          = 0.75
TileOverlap           = 0.25
RoiMargin             = 0.50
```

只有 Python 的定位和分类效果先确认正确以后，才进入 ONNX/C# 部署。

---

## 6. 导出通用 ONNX

两个 ONNX 不随产品变化，通常只需要导出一套。

```powershell
cd <repo-root>

python deploy\export_onnx.py --model patchcore --device cpu --exporter legacy --skip-verify
python deploy\export_onnx.py --model dinov2 --device cpu --exporter legacy --skip-verify
```

接口：

```text
patchcore_feature.onnx
  images      [B,3,320,320]
  memory_bank [M,1024]
  → patch_embeddings [B,1600,1024]
  → patch_scores     [B,1600]

dinov2_feature.onnx
  images [B,3,224,224]
  → cls_embedding    [B,384]
  → center_embedding [B,384]
```

> ONNX 是通用特征引擎。**不要把某个产品的 Memory/类别重新烘焙到 ONNX。**

---

## 7. 原 Python 产品模型 → C# 产品包

不要让 C# 重新生成 Memory Bank。

```powershell
python deploy\convert_python_product.py ^
  --product bottle ^
  --patchcore-model-dir industrial_anomaly\products\bottle\models\patchcore ^
  --defect-bank-dir industrial_anomaly\products\bottle\models\defect_bank ^
  --output-dir deploy\products\bottle ^
  --bbox-relative-threshold 0.78 ^
  --roi-margin 0.50 ^
  --copy-support-rois
```

转换关系：

```text
原 nnscorer_search_index.faiss
→ reconstruct 原 coreset vectors
→ patchcore_memory.bin

原 cls/embeddings.npz
→ defect_cls.bin

原 center/embeddings.npz
→ defect_center.bin
```

转换过程中不会重新训练、不会重新随机采样，也不会重新裁支持样本 ROI。

---

## 8. C# / WinForms 部署

编译：

```powershell
dotnet build deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -c Release
```

验证产品包：

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  validate-product deploy\models deploy\products\bottle
```

单图检测：

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  inspect deploy\models deploy\products\bottle data\bottle\test\Image_1.bmp deploy\outputs\Image_1_marked.jpg
```

批量检测：

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  inspect-folder deploy\models deploy\products\bottle data\bottle\test deploy\outputs\bottle_test
```

C# Runtime **不再包含产品训练/Memory 建模功能**。

---

## 9. WinForms 应该做什么

WinForms 只负责：

```text
产品模型导入 / 验证 / 激活
相机取图
检测开始 / 停止
最终 BBox
PatchCore Score
缺陷类别
Similarity / Margin
检测记录
系统设置
模型诊断
```

WinForms 不负责：

```text
PatchCore 训练
Greedy Coreset
FAISS Memory 构建
DINO defect bank 重建
```

训练和产品包转换在训练电脑完成。

---

## 10. 当前最重要的验收：Python ↔ ONNX parity

正式上线前必须对同一张固定 tile 逐级比较：

```text
1. input tensor
2. Patch embedding [1600,1024]
3. nearest-neighbor patch score [1600]
4. anomaly map
5. bbox
6. DINO CLS [384]
7. DINO Center [384]
8. similarity / final class
```

规则：

- 第 2 步不一致：先修 ONNX feature graph，不要调 BBox 参数。
- 第 2 步一致、第 3 步不一致：检查 squared-L2 与 Memory 输入。
- 第 3 步一致、BBox 不一致：检查插值、高斯、阈值、连通域、tile merge。
- DINO embedding 一致但分类不同：检查 bank 行顺序、labels、cosine 和融合权重。

当前 ONNX/C# 应视为**待 parity 完整验收的部署版本**，不能仅凭“能跑通”认定和 Python 1:1 等价。

---

## 11. 不要删除/修改的核心目录

```text
patchcore/
third_party_licenses/
```

`patchcore/` 是项目依赖的 PatchCore 核心实现；应用改动优先放在 `app/`、`industrial_anomaly/` 或 `deploy/`。

更详细的训练说明见 `industrial_anomaly/README.md`，部署说明见 `deploy/README.md`。
