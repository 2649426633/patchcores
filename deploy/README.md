# Generic ONNX Runtime + Dynamic C# Product Models

This deployment layer separates the **generic visual engines** from each
**product-specific model**.

```text
Generic engine (export once)
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json

C# product model (build dynamically)
└── <product>/
    ├── patchcore_memory.bin
    ├── defect_cls.bin
    ├── defect_center.bin
    ├── product_model.json
    └── support_rois/
```

No product name or defect class is hard-coded into the ONNX models. C# can
create `phone`, `glass`, `bottle`, or any later product by selecting a normal
folder and an arbitrary set of `class name -> image folder` mappings.

## Architecture

```text
C# normal images
    ↓ tiled full-image views
patchcore_feature.onnx
    ↓ [1600, 1024] patch embeddings / tile
C# builds product PatchCore memory
    ↓
patchcore_memory.bin

C# known-defect images
    ↓
PatchCore ONNX + product memory
    ↓ anomaly localization / ROI
DINOv2 ONNX
    ├── CLS [384]
    └── Center [384]
    ↓
defect_cls.bin + defect_center.bin + labels

New image
    ↓ tiled PatchCore ONNX + product memory
    ↓ final anomaly bbox
DINOv2 ONNX + product defect bank
    ↓
final known-defect candidate
```

## 1. Export the two generic ONNX models

Requirements already expected by the repository:

```text
D:\wlenai\weights\wide_resnet50_2-95faca4d.pth
D:\wlenai\weights\dinov2_vits14_pretrain.pth
D:\wlenai\third_party\dinov2\hubconf.py
```

Install the export-only packages:

```powershell
cd D:\wlenai
python -m pip install -r deploy\requirements.txt
```

Export and verify:

```powershell
python deploy\export_onnx.py --device cuda
```

If CUDA is not wanted for export:

```powershell
python deploy\export_onnx.py --device cpu
```

Generated locally (gitignored):

```text
D:\wlenai\deploy\models\
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json
```

`export_onnx.py` validates the ONNX files with ONNX Runtime unless
`--skip-verify` is explicitly supplied.

### PatchCore ONNX interface

```text
images       float32 [B, 3, 320, 320]
memory_bank  float32 [M, 1024]       # dynamic product memory

→ patch_embeddings float32 [B, 1600, 1024]
→ patch_scores     float32 [B, 1600]
```

`patch_scores` are minimum squared-L2 distances to the supplied memory bank,
matching the repository's FAISS `IndexFlatL2` distance definition.

### DINOv2 ONNX interface

```text
images float32 [B, 3, 224, 224]

→ cls_embedding    float32 [B, 384]
→ center_embedding float32 [B, 384]
```

Both DINOv2 outputs are L2-normalized. The center output uses the same 50%
center-patch pooling used by the Python production path.

## 2. Compile the C# runtime

The first implementation targets .NET 8 on Windows.

```powershell
cd D:\wlenai
dotnet build deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -c Release
```

The runtime project is:

```text
deploy\csharp\IndustrialAnomaly.Runtime\
```

Important classes:

```text
OnnxFeatureEngine          generic PatchCore + DINOv2 ONNX sessions
ProductModelBuilder        build a product from user-selected folders
ProductModel               load saved product banks and classify DINO features
PatchCoreTiledInspector    full-image tiled anomaly localization
IndustrialAnomalyEngine    end-to-end final inspection + final mark
```

## 3. Build a product dynamically from C#

Example with the current phone data:

```powershell
dotnet run --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- `
  build `
  D:\wlenai\deploy\models `
  D:\wlenai\deploy\products `
  phone `
  D:\wlenai\data\phone\good `
  shao1=D:\wlenai\data\phone\shao1 `
  shao2=D:\wlenai\data\phone\shao2 `
  shao3=D:\wlenai\data\phone\shao3
```

The class list is completely dynamic. For example, a future product could use:

```text
scratch=D:\data\new_product\scratch
crack=D:\data\new_product\crack
dirty=D:\data\new_product\dirty
```

without changing or re-exporting either ONNX file.

Output for phone:

```text
D:\wlenai\deploy\products\phone\
├── patchcore_memory.bin
├── defect_cls.bin
├── defect_center.bin
├── product_model.json
└── support_rois\
    ├── shao1\
    ├── shao2\
    └── shao3\
```

## 4. Inspect an image from C#

Without a calibrated PASS/NG threshold:

```powershell
dotnet run --project deploy\csharp\IndustrialAnomaly.Console\IndustrialAnomaly.Console.csproj -- `
  inspect `
  D:\wlenai\deploy\models `
  D:\wlenai\deploy\products\phone `
  D:\wlenai\data\phone\test\Image_1.bmp `
  D:\wlenai\deploy\outputs\Image_1_marked.jpg
```

The final marked image contains only the final primary bbox and final result,
not all intermediate R1/R2/R3 debug candidates.

When an independently calibrated anomaly threshold exists, append it as the
last argument:

```text
... Image_1_marked.jpg <threshold>
```

Do not tune a production PASS/NG threshold from the final test images.

## 5. WinForms integration

The Console project is only a validation harness. A WinForms UI can call the
runtime directly.

Dynamic product definition:

```csharp
using var featureEngine = new OnnxFeatureEngine(@"D:\wlenai\deploy\models");
var builder = new ProductModelBuilder(featureEngine);

var definition = new ProductBuildDefinition
{
    ProductName = "phone",
    NormalImageDirectory = @"D:\wlenai\data\phone\good",
    DefectClasses = new[]
    {
        new DefectClassDefinition
        {
            Name = "shao1",
            ImageDirectory = @"D:\wlenai\data\phone\shao1"
        },
        new DefectClassDefinition
        {
            Name = "shao2",
            ImageDirectory = @"D:\wlenai\data\phone\shao2"
        },
        new DefectClassDefinition
        {
            Name = "shao3",
            ImageDirectory = @"D:\wlenai\data\phone\shao3"
        }
    }
};

builder.Build(
    definition,
    @"D:\wlenai\deploy\products",
    message => Console.WriteLine(message)
);
```

Inference:

```csharp
using var featureEngine = new OnnxFeatureEngine(@"D:\wlenai\deploy\models");
var product = ProductModel.Load(@"D:\wlenai\deploy\products\phone");
var engine = new IndustrialAnomalyEngine(featureEngine, product);

var result = engine.InspectFile(
    @"D:\wlenai\data\phone\test\Image_1.bmp",
    @"D:\wlenai\deploy\outputs\Image_1_marked.jpg"
);
```

A WinForms layer only needs to collect:

```text
ProductName
NormalImageDirectory
List<DefectClassDefinition>
```

and pass them into `ProductModelBuilder`.

## Important parity note

The ONNX PatchCore feature graph mirrors the current Python feature pipeline:

```text
WideResNet50-2
→ layer2 + layer3
→ 3x3 patchify
→ layer3 spatial alignment to layer2
→ MeanMapper 1024
→ Aggregator 1024
```

Tiled localization, anomaly-map smoothing, connected-component evidence,
region ranking, IoU merge, ROI margin, DINO CLS/Center pooling and exemplar
cosine classification are also ported into the C# runtime.

The one intentional first-version difference is **normal-memory sampling**:

```text
Python production: ApproximateGreedyCoresetSampler
C# deployment v1: bounded streaming reservoir sampling
```

C# uses a bounded streaming strategy so a user can build models from large
high-resolution datasets without first holding every `[N,1024]` patch vector in
RAM. `product_model.json` records the strategy as `bounded_reservoir_v1`.

Therefore the C# runtime should be treated as a deployment baseline until the
Python-vs-ONNX-vs-C# parity test is run on the same normal/support/test images.
The next validation step is to compare PatchCore embeddings/scores, final bbox,
DINO embeddings and final class on the existing phone dataset.
