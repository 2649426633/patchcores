# SubspaceAD industrial DINOv2-S/14 workflow

这条路线用于工业部署评估，直接复用项目已有的本地 DINOv2-S/14：

```text
third_party/dinov2/
weights/dinov2_vits14_pretrain.pth
```

不需要下载约 4.5 GB 的 DINOv2-Giant，也不需要 Hugging Face 在线加载。

## 与论文/官方代码的关系

`third_party/subspacead/` 保持官方源码不修改。

本项目只增加适配层：

```text
app/subspacead/dinov2s_extractor.py
industrial_anomaly/subspacead_vits14.py
```

DINOv2-S 参数采用官方 SubspaceAD backbone ablation 中的轻量设置：

```text
backbone        = DINOv2-S/14
input           = 448 x 448
layers          = HF -4,-5
local blocks    = 7,8
aggregation     = mean
PCA EV          = 0.99
augmentation    = rotation
aug count       = 30
image score     = mean top 1%
map smoothing   = Gaussian sigma 4
```

这不是论文 DINOv2-Giant 主结果配置，因此不能把实际测试结果标成 Giant 主模型结果。

## 1. 先做最小测试

第一次先关闭增强，确认整条链路能运行：

```cmd
cd /d D:\wlenai
git pull --ff-only origin master

python industrial_anomaly\subspacead_vits14.py fit ^
  --product phone ^
  --normal-dir D:\wlenai\data\phone\good ^
  --shots 1 ^
  --aug-count 0
```

生成：

```text
industrial_anomaly/products/phone/models/subspacead_vits14/
├── pca_model.npz
└── config.json
```

然后检测：

```cmd
python industrial_anomaly\subspacead_vits14.py inspect ^
  --product phone ^
  --input D:\wlenai\data\phone\test\Image_3.bmp
```

输出默认在：

```text
industrial_anomaly/outputs/phone/subspacead_vits14/Image_3/
├── anomaly_map.npy
├── heatmap.jpg
├── overlay.jpg
└── result.json
```

`result.json` 会记录：

```text
image_score_top_1pct
patch_score_min
patch_score_max
bbox_original_image
inference_ms
device
```

当前 BBox 使用相对热图阈值，仅用于定位可视化，不是 PASS/NG 阈值。

## 2. 跑官方 DINOv2-S few-shot 设置

最小测试通过后再恢复 30 次旋转增强：

```cmd
python industrial_anomaly\subspacead_vits14.py fit ^
  --product phone ^
  --normal-dir D:\wlenai\data\phone\good ^
  --shots 4 ^
  --aug-count 30
```

再检测单张或整个文件夹：

```cmd
python industrial_anomaly\subspacead_vits14.py inspect ^
  --product phone ^
  --input D:\wlenai\data\phone\test
```

## 3. CPU 工业机测试

训练/拟合 PCA 可以在有 GPU 的电脑完成。产品 PCA 模型保存后，在目标工业电脑上测试 CPU 推理：

```cmd
python industrial_anomaly\subspacead_vits14.py inspect ^
  --product phone ^
  --input D:\wlenai\data\phone\test\Image_3.bmp ^
  --device cpu
```

重点查看 `result.json` 的 `inference_ms`。

正式部署阶段计划：

```text
DINOv2-S intermediate features
        ↓
ONNX Runtime
        ↓
PCA projection / reconstruction residual
        ↓
anomaly map
        ↓
BBox
        ↓
C# / WinForms
```

PCA 本身不需要 PyTorch、Python 或 FAISS。正式 PASS/NG 阈值必须用独立校准数据确定，不能从最终 test 数据反推。
