# PatchCore Industrial Anomaly Detection - Phase 1

This repository vendors the PatchCore core implementation and adds a small application layer for offline training and single-image inference.

## Required local files

Do **not** upload datasets, generated FAISS indexes, or model weights to GitHub.

Place the WideResNet50-2 ImageNet weight at exactly:

```text
weights/wide_resnet50_2-95faca4d.pth
```

For MVTec AD screw, keep the dataset locally as:

```text
data/screw/
├─ train/good/
├─ test/
└─ ground_truth/
```

## Project structure

```text
patchcores/
├─ app/
│  └─ anomaly/
│     ├─ image_dataset.py
│     ├─ preprocessing.py
│     ├─ patchcore_adapter.py
│     └─ postprocessing.py
├─ patchcore/                 # vendored PatchCore core
├─ weights/
│  └─ wide_resnet50_2-95faca4d.pth   # local only, ignored by git
├─ data/                      # local only, ignored by git
├─ products/                  # generated model files, ignored by git
├─ outputs/                   # generated visualizations, ignored by git
├─ train_patchcore.py
├─ predict_patchcore.py
└─ requirements_project.txt
```

## Environment

Install PyTorch and torchvision first using versions appropriate for your CPU/CUDA environment. Then install application dependencies:

```powershell
python -m pip install -r requirements_project.txt
```

On the current Windows environment, if FAISS and PyTorch trigger an OpenMP duplicate-runtime error during development, the temporary workaround used in this project session is:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

Do not treat that environment variable as a final production deployment solution.

## Train screw PatchCore

```powershell
python train_patchcore.py
```

Defaults:

```text
normal images: data/screw/train/good
model output:  products/screw/models/patchcore
```

Successful training creates locally:

```text
products/screw/models/patchcore/
├─ patchcore_params.pkl
└─ nnscorer_search_index.faiss
```

## Predict one image

```powershell
python predict_patchcore.py "data/screw/test/scratch_head/000.png"
```

Outputs include:

- image-level anomaly score
- anomaly heatmap
- heatmap overlay
- candidate anomaly bounding box

The current phase deliberately does not assign PASS/NG because a product-specific anomaly threshold has not yet been calibrated.

## Offline behavior

`PatchCoreAdapter` does not use the upstream `load_from_path()` backbone reconstruction path. Both training and inference load WideResNet50-2 from the local file in `weights/`, so prediction does not need to download torchvision weights.

## Third-party code

The vendored PatchCore core originates from Amazon Science's `patchcore-inspection` repository and is distributed under Apache License 2.0. See `third_party_licenses/patchcore/`.
