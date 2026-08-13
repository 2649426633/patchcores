# Industrial Anomaly — Clean Main Project

这个目录只保留当前正式主流程：

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

不包含当前非核心实验：Unknown、LODO、Prototype、KMeans、PatchMatch、Dynamic Selector 等。

## 1. 目录

本地建议直接使用：

```text
D:\wlenai\industrial_anomaly\
│
├── train_patchcore.py
├── build_defect_bank.py
├── inspect_image.py
│
├── products\
│   └── <product>\
│       ├── train\
│       │   └── good\
│       ├── defects\
│       │   ├── defect_A\
│       │   ├── defect_B\
│       │   └── ...
│       └── models\
│           ├── patchcore\
│           └── defect_bank\
│               ├── cls\
│               └── center\
│
└── outputs\
```

底层已经验证的 PatchCore/DINOv2 adapter、官方 PatchCore 黑盒、权重仍复用仓库根目录：

```text
D:\wlenai\app\
D:\wlenai\patchcore\
D:\wlenai\weights\
D:\wlenai\third_party\dinov2\
```

这样避免复制两份核心实现造成版本漂移。你日常测试只需要进入 `industrial_anomaly`。

## 2. 新产品数据准备

例如测试 MVTec `bottle`：

```text
products\bottle\
├── train\good\
└── defects\
    ├── broken_large\
    ├── broken_small\
    └── contamination\
```

规则：

- `train/good`：只放正常图。PatchCore 无监督训练只读取这里。
- `defects/<class>`：放该已知异常类别的完整图片，推荐每类约 5–10 张有代表性的图片。
- 不需要 GT mask。
- DINOv2 不微调，不训练分类头；只是建立 exemplar feature bank。

也可以不复制数据，直接通过 `--normal-dir` / `--defects-dir` 指向原数据集目录。

## 3. PatchCore 无监督训练

在：

```powershell
cd D:\wlenai\industrial_anomaly
```

如果已经把数据复制到 clean project：

```powershell
python train_patchcore.py --product bottle
```

如果继续使用原 MVTec 数据目录：

```powershell
python train_patchcore.py --product bottle --normal-dir ..\data\bottle\train\good
```

默认正式配置：

```text
imagesize = 320
resize    = 366
layers    = layer2 + layer3
coreset   = 0.10
```

模型保存到：

```text
products\bottle\models\patchcore\
```

PatchCore 训练阶段不使用任何缺陷标签或缺陷图片。

## 4. 建立已知异常库

如果数据已经整理到 `products/bottle/defects`：

```powershell
python build_defect_bank.py --product bottle --shots 10
```

直接使用原 MVTec test 缺陷目录也可以：

```powershell
python build_defect_bank.py --product bottle --defects-dir ..\data\bottle\test --shots 10
```

脚本会自动忽略 `good` 目录。

当前正式 Known 分类特征：

```text
PatchCore ROI
  ↓
DINOv2 CLS              -> exemplar bank
DINOv2 Patch Center 50% -> exemplar bank
  ↓
0.5 * CLS class score + 0.5 * Center class score
```

Bank 保存到：

```text
products\bottle\models\defect_bank\
├── cls\
├── center\
├── support_rois\
├── support_overlays\
└── bank_config.json
```

如果某类不足 10 张，可使用：

```powershell
python build_defect_bank.py --product bottle --shots 5
```

或者全部使用：

```powershell
python build_defect_bank.py --product bottle --shots 0
```

## 5. 测试一张异常图片

```powershell
python inspect_image.py D:\wlenai\data\bottle\test\broken_large\000.png --product bottle
```

输出：

```text
PatchCore anomaly score
bbox
bbox source
known defect class
Top-1 similarity
Top-2 class
margin
```

并保存：

```text
outputs\bottle\<image_name>\
├── roi.png
├── bbox.jpg
├── anomaly_map.png
└── result.json
```

## 6. 关于 PASS / NG

目前不在这个 clean project 里写死 MVTec test-derived PASS/NG threshold。

默认运行 `inspect_image.py` 时：

- 输出 PatchCore anomaly score；
- 自动定位异常区域；
- 输出已知异常分类候选；
- 不擅自用测试标签生成 PASS/NG 阈值。

如果以后已经用独立 calibration set 得到正式阈值，可显式传入：

```powershell
python inspect_image.py test.png --product bottle --anomaly-threshold <YOUR_CALIBRATED_THRESHOLD>
```

## 7. 换其他类别

例如：

```text
bottle
cable
capsule
hazelnut
metal_nut
pill
zipper
```

只需要换 `--product` 和数据目录，不需要修改算法代码。

标准流程始终是：

```powershell
python train_patchcore.py --product <name> --normal-dir <normal_images>
python build_defect_bank.py --product <name> --defects-dir <known_defect_classes> --shots 10
python inspect_image.py <test_image> --product <name>
```
