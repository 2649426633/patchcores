# Generic ONNX Runtime + Original Python Product Banks

The deployment architecture is now intentionally split into two parts:

```text
Generic ONNX engine (export once)
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json

Product model (must come from the original Python-trained model)
└── <product>/
    ├── patchcore_memory.bin
    ├── defect_cls.bin
    ├── defect_center.bin
    ├── product_model.json
    └── conversion_report.json
```

**Production C# no longer rebuilds PatchCore memory or DINO exemplar banks.**
The previous `bounded_reservoir_v1` C# rebuild changed PatchCore's normal feature
space and therefore could change anomaly scores, anomaly maps, BBoxes, ROIs and
final DINO classifications.

The required production flow is:

```text
Training machine / Python
  original PatchCore
  → ApproximateGreedyCoresetSampler
  → original FAISS memory

  original DINO bank
  → cls/embeddings.npz
  → center/embeddings.npz
  → metadata.json
            ↓
  deploy/convert_python_product.py
            ↓
  patchcore_memory.bin
  defect_cls.bin
  defect_center.bin
  product_model.json
            ↓
────────────────────────────────────────────
Deployment machine / C# / WinForms
  Python:      NOT REQUIRED
  PyTorch:     NOT REQUIRED
  FAISS:       NOT REQUIRED
  ONNX Runtime + OpenCvSharp only
```

## 1. Generic ONNX files

Generated files:

```text
deploy/models/
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json
```

### PatchCore ONNX interface

```text
images       float32 [B, 3, 320, 320]
memory_bank  float32 [M, 1024]

→ patch_embeddings float32 [B, 1600, 1024]
→ patch_scores     float32 [B, 1600]
```

`memory_bank` is supplied at runtime from `patchcore_memory.bin` and must be the
memory from the original Python FAISS index.

### DINOv2 ONNX interface

```text
images float32 [B, 3, 224, 224]

→ cls_embedding    float32 [B, 384]
→ center_embedding float32 [B, 384]
```

## 2. Convert the original Python product model

The converter does **not** retrain, resample or re-embed images.

PatchCore:

```text
nnscorer_search_index.faiss
→ reconstruct the vectors actually stored in FAISS
→ verify reconstructed vectors reproduce FAISS nearest-neighbour L2 results
→ patchcore_memory.bin
```

DINOv2:

```text
defect_bank/cls/embeddings.npz
→ defect_cls.bin

defect_bank/center/embeddings.npz
→ defect_center.bin

metadata.json
→ DefectLabels / Classes in product_model.json
```

Example for the existing phone model:

```powershell
cd D:\wlenai
python deploy\convert_python_product.py `
  --product phone `
  --patchcore-model-dir D:\wlenai\industrial_anomaly\products\phone\models\patchcore `
  --defect-bank-dir D:\wlenai\industrial_anomaly\products\phone\models\defect_bank `
  --output-dir D:\wlenai\deploy\products\phone `
  --bbox-relative-threshold 0.78 `
  --roi-margin 0.50 `
  --copy-support-rois
```

Output:

```text
deploy/products/phone/
├── patchcore_memory.bin
├── defect_cls.bin
├── defect_center.bin
├── product_model.json
├── conversion_report.json
└── support_rois/              # optional
```

`product_model.json` records:

```text
ProductModelSource        = python_export
PatchCoreMemoryStrategy   = python_faiss_memory_exact
```

The C# runtime rejects the deprecated `bounded_reservoir*` strategy.

## 3. Compile C# runtime

```powershell
dotnet build deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -c Release
```

Runtime classes:

```text
OnnxFeatureEngine          generic ONNX sessions
ProductModel               loads the converted Python banks
PatchCoreTiledInspector    tiled localization
IndustrialAnomalyEngine    PatchCore → ROI → DINO → final result
```

`ProductModelBuilder` is legacy development code and must not be used for
production model creation.

## 4. Validate the converted product

```powershell
dotnet run --no-build -c Release `
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- `
  validate-product `
  D:\wlenai\deploy\models `
  D:\wlenai\deploy\products\phone
```

Expected fields include:

```text
PRODUCT MODEL VALID
source=python_export
memory_strategy=python_faiss_memory_exact
patchcore_memory=<rows>x1024
defect_cls=<N>x384
defect_center=<N>x384
classes=...
```

## 5. Inspect from C#

```powershell
dotnet run --no-build -c Release `
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- `
  inspect `
  D:\wlenai\deploy\models `
  D:\wlenai\deploy\products\phone `
  D:\wlenai\data\phone\test\Image_1.bmp `
  D:\wlenai\deploy\outputs\Image_1_marked.jpg
```

Folder inspection:

```powershell
dotnet run --no-build -c Release `
  --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- `
  inspect-folder `
  D:\wlenai\deploy\models `
  D:\wlenai\deploy\products\phone `
  D:\wlenai\data\phone\test `
  D:\wlenai\deploy\outputs\phone_test
```

## 6. WinForms integration

WinForms must **load an already converted product model**. It must not rebuild
PatchCore memory from normal images and must not rebuild DINO banks from support
images.

```csharp
using var featureEngine = new OnnxFeatureEngine(@"D:\App\engine");
var product = ProductModel.Load(@"D:\App\products\phone");
var engine = new IndustrialAnomalyEngine(featureEngine, product);

var result = engine.InspectFile(
    imagePath,
    markedOutputPath,
    anomalyThreshold: null
);
```

Recommended product-management UI:

```text
Import product model folder
  ↓
validate ProductModelSource == python_export
  ↓
validate memory strategy == python_faiss_memory_exact
  ↓
validate dimensions against engine_config.json
  ↓
activate product
```

Do not expose a production button that rebuilds memory/banks locally.

## 7. Parity gate before WinForms production acceptance

Using the original Python model and the converted product banks, compare the
same fixed 320x320 tile step-by-step:

```text
1. normalized input tensor
2. PatchCore patch embedding [1600,1024]
3. nearest-neighbour patch score [1600]
4. anomaly map
5. BBox
6. DINO CLS [384]
7. DINO Center [384]
8. similarity / class
```

The ONNX feature graph is still considered **unaccepted for production** until
this parity test is completed. Preserving the original Python memory/banks
removes the known C# reservoir mismatch, but it does not by itself prove the
hand-rewritten PatchCore ONNX feature graph is numerically equivalent.

PASS/NG threshold calibration is a separate production-validation task and
must not be tuned on the final test set.
