# Industrial Anomaly — Clean Main Project

这个目录只保留正式主流程：

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

## 1. 固定参考路径

假设你的仓库位于：

```text
D:\wlenai\
```

代码固定使用两个根路径：

```text
REPO_ROOT  = D:\wlenai
CLEAN_ROOT = D:\wlenai\industrial_anomaly
```

底层核心和本地模型资产参考：

```text
D:\wlenai\app\
D:\wlenai\patchcore\
D:\wlenai\weights\wide_resnet50_2-95faca4d.pth
D:\wlenai\weights\dinov2_vits14_pretrain.pth
D:\wlenai\third_party\dinov2\
D:\wlenai\third_party_licenses\
```

正式业务数据和输出参考：

```text
D:\wlenai\industrial_anomaly\products\<product>\
D:\wlenai\industrial_anomaly\outputs\<product>\
```

### 相对路径规则

三个入口脚本现在统一规定：

> 所有命令行中的相对路径，都以 `D:\wlenai\industrial_anomaly\` 为参考目录解析，而不是以当前终端所在目录解析。

因此下面两种运行方式含义一致：

```powershell
cd D:\wlenai\industrial_anomaly
python train_patchcore.py --product bottle --normal-dir ..\data\bottle\train\good
```

以及：

```powershell
cd D:\wlenai
python industrial_anomaly\train_patchcore.py --product bottle --normal-dir ..\data\bottle\train\good
```

两者都会读取：

```text
D:\wlenai\data\bottle\train\good
```

绝对路径则保持原样，例如：

```powershell
--normal-dir D:\datasets\mvtec\bottle\train\good
```

## 2. Clean Project 目录

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

## 3. 新产品数据准备

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

- `train/good`：只放正常图，PatchCore 无监督训练只读取这里。
- `defects/<class>`：放该已知异常类别的完整图片，推荐每类约 5–10 张有代表性的图片。
- 不需要 GT mask。
- DINOv2 不微调，不训练分类头，只建立 exemplar feature bank。

也可以不复制数据，直接通过 `--normal-dir` / `--defects-dir` 指向原数据集目录。

## 4. PatchCore 无监督训练

如果数据已经复制到 clean project：

```powershell
python train_patchcore.py --product bottle
```

默认读取：

```text
D:\wlenai\industrial_anomaly\products\bottle\train\good
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

模型默认保存到：

```text
D:\wlenai\industrial_anomaly\products\bottle\models\patchcore\
```

PatchCore 训练阶段不使用任何缺陷标签或缺陷图片。

## 5. 建立已知异常库

如果数据位于 clean project：

```powershell
python build_defect_bank.py --product bottle --shots 10
```

默认读取：

```text
D:\wlenai\industrial_anomaly\products\bottle\defects\
```

直接使用原 MVTec test 目录：

```powershell
python build_defect_bank.py --product bottle --defects-dir ..\data\bottle\test --shots 10
```

脚本会自动忽略 `good` 目录。

当前正式 Known 分类特征：

```text
PatchCore ROI
  ↓
DINOv2 CLS          -> exemplar bank
DINOv2 Patch Center -> exemplar bank
  ↓
0.5 * CLS class score + 0.5 * Center class score
```

Bank 默认保存到：

```text
D:\wlenai\industrial_anomaly\products\bottle\models\defect_bank\
├── cls\
├── center\
├── support_rois\
├── support_overlays\
└── bank_config.json
```

如果某类不足 10 张，可以使用：

```powershell
python build_defect_bank.py --product bottle --shots 5
```

或者全部使用：

```powershell
python build_defect_bank.py --product bottle --shots 0
```

## 6. 测试一张图片

相对路径：

```powershell
python inspect_image.py ..\data\bottle\test\broken_large\000.png --product bottle
```

对应实际输入：

```text
D:\wlenai\data\bottle\test\broken_large\000.png
```

也可以直接使用绝对路径：

```powershell
python inspect_image.py D:\wlenai\data\bottle\test\broken_large\000.png --product bottle
```

默认输出：

```text
D:\wlenai\industrial_anomaly\outputs\bottle\000\
├── roi.png
├── bbox.jpg
├── anomaly_map.png
└── result.json
```

## 7. 关于 PASS / NG

当前 clean project 不写死任何来自 MVTec test 标签的 PASS/NG threshold。

默认 `inspect_image.py`：

- 输出 PatchCore anomaly score；
- 自动定位异常区域；
- 输出已知异常分类候选；
- 不擅自用测试标签生成 PASS/NG 阈值。

以后使用独立 calibration set 得到正式阈值后，可以显式传入：

```powershell
python inspect_image.py test.png --product bottle --anomaly-threshold <YOUR_CALIBRATED_THRESHOLD>
```

## 8. 标准三步流程

```powershell
python train_patchcore.py --product <name> --normal-dir <normal_images>
python build_defect_bank.py --product <name> --defects-dir <known_defect_classes> --shots 10
python inspect_image.py <test_image> --product <name>
```

换 `bottle / cable / capsule / hazelnut / metal_nut / pill / zipper` 等产品时，只替换 `--product` 和数据路径，不修改算法代码。
