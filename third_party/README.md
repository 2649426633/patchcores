# Local third-party source

此目录用于放本地依赖源码，源码本体不提交到本仓库。

当前需要：

```text
third_party/
└── dinov2/
    ├── hubconf.py
    └── dinov2/
```

Python 代码通过本地路径加载 DINOv2，不从网络下载。

对应权重放在：

```text
weights/dinov2_vits14_pretrain.pth
```

`third_party/dinov2/` 已被 `.gitignore` 忽略。
