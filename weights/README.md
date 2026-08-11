# Local Backbone Weights

Place the WideResNet50-2 ImageNet V1 checkpoint here:

```text
weights/wide_resnet50_2-95faca4d.pth
```

The `.pth` file is intentionally ignored by Git and is not stored in this repository.

Both training and prediction use this local file through `app/anomaly/patchcore_adapter.py`, so the application does not need to download torchvision backbone weights at runtime.
