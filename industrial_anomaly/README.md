# Industrial Anomaly — Clean Main Project

正式主流程：

```text
正常图 -> PatchCore 无监督训练
             ↓
新图 -> PatchCore anomaly regions / ROI
             ↓
       Frozen DINOv2
        ├─ CLS
        └─ Patch Center
             ↓
       Exemplar Bank
             ↓
       已知异常类别
```

仓库根目录假设为 `D:\wlenai`，日常入口位于：

```text
D:\wlenai\industrial_anomaly\
├── train_patchcore.py
├── build_defect_bank.py
├── inspect_image.py
└── inspect_folder.py
```

所有命令行相对路径都固定以 `D:\wlenai\industrial_anomaly\` 为参考目录。

## Phone 高分辨率数据

```text
D:\wlenai\data\phone\
├── good\      # 50 正常
├── shao1\     # 10 已知异常
├── shao2\     # 10 已知异常
├── shao3\     # 10 已知异常
└── test\      # 6 独立测试
```

### 为什么 phone 使用 tiled 模式

phone 原图约 5472x3648。旧的 `Resize(366) -> CenterCrop(320)` 只观察原图中间区域，左右各约 1100px 完全不会进入 PatchCore，且单 bbox 后处理会丢掉有效的次级异常区域。

`tiled` 模式改为：

```text
完整原图
  ↓
多个重叠方形 tile（默认短边 75%，25% overlap）
  ↓
每个 tile 直接 resize 到 320x320
  ↓
PatchCore 分别推理
  ↓
所有异常连通域映射回原图
  ↓
重叠区域合并 / 去重
  ↓
R1 / R2 / R3 ... 多异常区域
```

这样既覆盖整图，又比“整图直接压缩到 320”保留更多小缺陷细节。

## Phone 三步命令

先同步最新代码：

```powershell
cd D:\wlenai
git fetch origin
git pull --ff-only origin master
cd D:\wlenai\industrial_anomaly
```

### 1. 重新训练 tiled PatchCore

```powershell
python train_patchcore.py --product phone --normal-dir ..\data\phone\good --mode tiled
```

默认：

```text
imagesize      = 320
tile_fraction  = 0.75
tile_overlap   = 0.25
layers         = layer2 + layer3
coreset        = 0.10
```

训练完成后模型目录会额外保存：

```text
products\phone\models\patchcore\inspection_config.json
```

后续建库和测试会自动读取 `tiled` 模式与 tile 参数。

### 2. 重新建立 shao1 / shao2 / shao3 bank

```powershell
python build_defect_bank.py --product phone --defects-dir ..\data\phone --classes shao1 shao2 shao3 --shots 10
```

每张 support 图使用 tiled PatchCore 全图定位，取主异常 ROI 建立 DINOv2 exemplar bank。

### 3. 批量测试 6 张 test

```powershell
python inspect_folder.py ..\data\phone\test --product phone
```

输出：

```text
D:\wlenai\industrial_anomaly\outputs\phone\test\
├── results.csv
├── results.json
├── marked\
│   └── *_marked.jpg          # 原始完整图 + R1/R2/R3 多框 + 类别
├── rois\
│   └── *_R1_roi.png ...      # 每个 PatchCore 候选区域
├── full_heatmaps\
│   └── *_heatmap.jpg         # 完整原图 tiled heatmap
└── anomaly_maps\
```

每个候选区域都会单独输出：

```text
PatchCore region
bbox（原始图坐标）
shao1 / shao2 / shao3
DINOv2 similarity
margin
```

`results.csv/json` 会保留 `all_regions`，不再只保留一个 bbox。

## 定位调试顺序

如果结果仍不正确，先看：

```text
outputs\phone\test\full_heatmaps\
```

- 热图在真实缺陷上，但框不对：后处理/合并问题。
- 热图完全不在真实缺陷上：PatchCore 特征或 normal 数据覆盖问题。
- 热图和框都对，但类别错：再处理 DINOv2 分类，不要先调分类器。

## PASS / NG

目前不从 test 标签生成生产 PASS/NG threshold。`--bbox-relative-threshold` 只是定位热区阈值，不是 PASS/NG 阈值。
