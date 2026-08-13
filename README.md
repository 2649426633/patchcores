# Industrial Anomaly Detection

Clean production-oriented pipeline for:

1. **Unsupervised anomaly detection/localization** with PatchCore trained on normal images only.
2. **Known defect classification** with frozen DINOv2 and a few-shot exemplar bank.

## Start here

All day-to-day commands are in:

```text
industrial_anomaly/
├── train_patchcore.py
├── build_defect_bank.py
├── inspect_image.py
└── README.md
```

The remaining repository folders are supporting core code:

```text
app/
├── anomaly/   # PatchCore adapter, preprocessing, localization
└── defect/    # DINOv2 adapter, exemplar bank, end-to-end ROI pipeline

patchcore/             # vendored PatchCore core; keep unchanged
weights/               # local weight instructions/placeholders
third_party_licenses/  # third-party licenses
```

## Typical workflow

From the repository root:

```powershell
cd industrial_anomaly
```

Train PatchCore from normal images only:

```powershell
python train_patchcore.py --product bottle --normal-dir ..\data\bottle\train\good
```

Build a known-defect exemplar bank:

```powershell
python build_defect_bank.py --product bottle --defects-dir ..\data\bottle\test --shots 10
```

Inspect one image:

```powershell
python inspect_image.py ..\data\bottle\test\broken_large\000.png --product bottle
```

See `industrial_anomaly/README.md` for the full folder layout and options.

## Local dependencies

Install `torch` and `torchvision` separately so they match your CUDA environment, then:

```powershell
pip install -r requirements.txt
```

Offline runtime also expects local model assets described in `weights/` and a local DINOv2 checkout at `third_party/dinov2/`.
