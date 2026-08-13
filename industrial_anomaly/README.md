# Industrial Anomaly — Clean Main Project

正式主流程：

```text
正常图 -> PatchCore 无监督训练
             ↓
新图 -> PatchCore anomaly map / bbox / ROI
             ↓
       Frozen DINOv2
        ├─ CLS
        └─ Patch Center
             ↓
     Real Exemplar Bank
             ↓
       已知异常类别
```

## 固定参考路径

假设仓库位于：

```text
D:\wlenai\
```

```text
REPO_ROOT  = D:\wlenai
CLEAN_ROOT = D:\wlenai\industrial_anomaly
```

运行时资产：

```text
D:\wlenai\app\
D:\wlenai\patchcore\
D:\wlenai\weights\wide_resnet50_2-95faca4d.pth
D:\wlenai\weights\dinov2_vits14_pretrain.pth
D:\wlenai\third_party\dinov2\
```

所有命令行相对路径都固定以 `D:\wlenai\industrial_anomaly\` 为参考目录解析。

## 正式入口

```text
industrial_anomaly\
├── train_patchcore.py
├── build_defect_bank.py
├── inspect_image.py
└── inspect_folder.py
```

- `train_patchcore.py`：正常图片训练 PatchCore。
- `build_defect_bank.py`：PatchCore 自动定位已知异常 ROI，建立 DINOv2 exemplar bank。
- `inspect_image.py`：测试单张图片。
- `inspect_folder.py`：批量测试整个目录，一次加载模型，并输出完整原图标记和 CSV/JSON。

## Phone 数据集

当前 phone 数据可以直接保持：

```text
D:\wlenai\data\phone\
├── good\      # 50 张正常图
├── shao1\     # 10 张已知异常
├── shao2\     # 10 张已知异常
├── shao3\     # 10 张已知异常
└── test\      # 6 张待测试异常图
```

不需要移动或复制数据。

### 1. PatchCore 无监督训练

```powershell
cd D:\wlenai\industrial_anomaly
python train_patchcore.py --product phone --normal-dir ..\data\phone\good
```

PatchCore 只读取 `good`，不会使用 shao1/shao2/shao3/test。

默认配置：

```text
imagesize = 320
resize    = 366
layers    = layer2 + layer3
coreset   = 0.10
```

模型保存：

```text
D:\wlenai\industrial_anomaly\products\phone\models\patchcore\
```

### 2. 建立 shao1 / shao2 / shao3 已知异常库

因为 `D:\wlenai\data\phone` 同时还有 `good` 和 `test`，必须显式指定三个异常类别：

```powershell
python build_defect_bank.py --product phone --defects-dir ..\data\phone --classes shao1 shao2 shao3 --shots 10
```

Bank 保存：

```text
D:\wlenai\industrial_anomaly\products\phone\models\defect_bank\
├── cls\
├── center\
├── support_rois\
├── support_overlays\
└── bank_config.json
```

### 3. 批量测试 6 张 test 图片

```powershell
python inspect_folder.py ..\data\phone\test --product phone
```

模型只加载一次，然后连续处理 test 中所有图片。

输出：

```text
D:\wlenai\industrial_anomaly\outputs\phone\test\
├── results.csv
├── results.json
├── marked\
│   ├── <image1>_marked.jpg
│   ├── <image2>_marked.jpg
│   └── ...
├── rois\
│   └── *_roi.png
└── anomaly_maps\
    └── *_anomaly.png
```

`marked` 图片是**原始完整分辨率图片**，不是 320x320 PatchCore crop。图片上会显示：

```text
异常 bbox
预测类别 shao1 / shao2 / shao3
PatchCore anomaly score
DINOv2 Top-1 similarity
```

`results.csv` 每张图片包含：

```text
image
patchcore_anomaly_score
bbox_source
crop_bbox_*
original_bbox_*
predicted_known_defect
top1_similarity
top2_class
top2_similarity
margin
marked_image
roi_image
anomaly_map
```

### 4. 单张测试

```powershell
python inspect_image.py ..\data\phone\test\001.jpg --product phone
```

单图输出包含：

```text
full_marked.jpg   # 原始完整图片 + bbox/标签
bbox_crop.jpg     # PatchCore 320 crop 坐标图
roi.png
anomaly_map.png
result.json
```

## PASS / NG 阈值

当前代码不会从 test 标签偷偷生成 PASS/NG threshold。默认输出 PatchCore anomaly score 和已知异常分类结果。

以后如果有独立 calibration set，可以显式传：

```powershell
python inspect_folder.py ..\data\phone\test --product phone --anomaly-threshold <CALIBRATED_THRESHOLD>
```

没有独立校准阈值时，不应把某个测试集分数硬写成生产 PASS/NG threshold。
