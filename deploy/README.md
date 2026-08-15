# ONNX / C# deployment

部署端只负责推理，不负责训练。

最终结构：

```text
Generic ONNX engine
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json

Converted product package
└── <product>/
    ├── patchcore_memory.bin
    ├── defect_cls.bin
    ├── defect_center.bin
    ├── product_model.json
    └── conversion_report.json
```

正式产品 bank 必须来自原 Python 已训练模型。

---

## 1. 为什么 C# 不再建产品模型

原 Python PatchCore：

```text
normal images
→ WideResNet50-2
→ patch embeddings
→ ApproximateGreedyCoresetSampler
→ FAISS IndexFlatL2
```

旧 C# 曾经使用 `bounded_reservoir` 重新采样正常特征，这会改变正常空间，从而改变：

```text
PatchCore score
→ anomaly map
→ bbox
→ ROI
→ DINO class
```

因此旧的 C# `ProductModelBuilder` 已从仓库删除。

生产端只允许加载：

```text
ProductModelSource      = python_export
PatchCoreMemoryStrategy = python_faiss_memory_exact
```

`ProductModel.Load()` 会拒绝旧 `bounded_reservoir*` 产品包。

---

## 2. 通用 ONNX

导出：

```powershell
python deploy\export_onnx.py --model patchcore --device cpu --exporter legacy --skip-verify
python deploy\export_onnx.py --model dinov2 --device cpu --exporter legacy --skip-verify
```

生成到：

```text
deploy/models/
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json
```

PatchCore：

```text
images       float32 [B,3,320,320]
memory_bank  float32 [M,1024]

→ patch_embeddings [B,1600,1024]
→ patch_scores     [B,1600]
```

DINOv2：

```text
images float32 [B,3,224,224]

→ cls_embedding    [B,384]
→ center_embedding [B,384]
```

---

## 3. 转换原 Python 产品模型

示例：

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

转换器不会重新训练、重新采样或重新跑 support 图片。

PatchCore：

```text
nnscorer_search_index.faiss
→ reconstruct FAISS 中实际保存的 coreset vectors
→ 验证 nearest-neighbour squared-L2
→ patchcore_memory.bin
```

DINO：

```text
cls/embeddings.npz
→ defect_cls.bin

center/embeddings.npz
→ defect_center.bin

metadata.json
→ labels / classes
```

输出：

```text
deploy/products/<product>/
├── patchcore_memory.bin
├── defect_cls.bin
├── defect_center.bin
├── product_model.json
├── conversion_report.json
└── support_rois/
```

---

## 4. C# Runtime

```text
deploy/csharp/
├── IndustrialAnomaly.Runtime/
│   ├── BinaryMatrix.cs
│   ├── ModelContracts.cs
│   ├── OnnxFeatureEngine.cs
│   ├── ProductModel.cs
│   ├── PatchCoreTiledInspector.cs
│   └── IndustrialAnomalyEngine.cs
└── IndustrialAnomaly.Console/
    └── Program.cs
```

职责：

```text
OnnxFeatureEngine       加载两个 ONNX
ProductModel            加载原 Python 转换出的三个 bank
PatchCoreTiledInspector tiled 定位
IndustrialAnomalyEngine PatchCore → ROI → DINO → 最终结果
```

编译：

```powershell
dotnet build deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -c Release
```

---

## 5. 验证产品包

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  validate-product deploy\models deploy\products\phone
```

应该看到：

```text
PRODUCT MODEL VALID
source=python_export
memory_strategy=python_faiss_memory_exact
patchcore_memory=<rows>x1024
defect_cls=<N>x384
defect_center=<N>x384
classes=...
```

---

## 6. C# 检测

单图：

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  inspect deploy\models deploy\products\phone data\phone\test\Image_1.bmp deploy\outputs\Image_1_marked.jpg
```

文件夹：

```powershell
dotnet run --no-build -c Release ^
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- ^
  inspect-folder deploy\models deploy\products\phone data\phone\test deploy\outputs\phone_test
```

---

## 7. WinForms integration

WinForms 只导入、验证、激活产品包：

```csharp
using var featureEngine = new OnnxFeatureEngine(engineDirectory);
var product = ProductModel.Load(productDirectory);
var engine = new IndustrialAnomalyEngine(featureEngine, product);

var result = engine.InspectFile(
    imagePath,
    markedOutputPath,
    anomalyThreshold: null
);
```

WinForms 产品管理建议：

```text
导入产品包
→ validate ProductModelSource
→ validate MemoryStrategy
→ validate dimensions
→ 激活产品
```

不要提供“现场训练 PatchCore / 重建 Memory / 重建 DINO Bank”按钮。

---

## 8. Parity gate

正式验收前，对同一固定输入逐级比较原 Python 与 ONNX/C#：

```text
1. normalized input tensor
2. Patch embedding [1600,1024]
3. nearest-neighbour patch score [1600]
4. anomaly map
5. bbox
6. DINO CLS [384]
7. DINO Center [384]
8. similarity / final class
```

第 2 步不通过时，不要继续靠调 bbox threshold 修结果。

当前部署代码应视为“已跑通、待完整 parity 验收”，而不是已经证明与 Python 1:1 等价。
