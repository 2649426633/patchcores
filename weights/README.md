# Local Model Assets

运行时使用本地模型文件，不依赖在线下载。

假设仓库根目录是：

```text
D:\wlenai\
```

请准备以下文件：

```text
D:\wlenai\weights\
├── wide_resnet50_2-95faca4d.pth
└── dinov2_vits14_pretrain.pth
```

以及本地 DINOv2 官方源码：

```text
D:\wlenai\third_party\dinov2\
├── hubconf.py
└── dinov2\
```

代码中的参考关系：

```text
app/anomaly/patchcore_adapter.py
  -> <repo_root>/weights/wide_resnet50_2-95faca4d.pth

app/defect/dinov2_adapter.py
  -> <repo_root>/third_party/dinov2
  -> <repo_root>/weights/dinov2_vits14_pretrain.pth
```

这些路径都基于 Python 文件自身位置推导 `<repo_root>`，因此不依赖当前终端工作目录。

`.pth` 权重以及 `third_party/dinov2/` 本地源码被 Git 忽略，不会上传到仓库。官方 PatchCore 核心目录 `patchcore/` 保持不修改。
