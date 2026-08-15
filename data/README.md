# Local datasets

此目录只放本地训练/测试数据，图片本身不要提交到 GitHub。

推荐每个产品使用统一结构：

```text
data/
└── <product>/
    ├── good/          # 正常图，只用于 PatchCore 无监督训练
    ├── <defect_a>/    # 已知缺陷类别 A，供 DINO exemplar bank
    ├── <defect_b>/    # 已知缺陷类别 B
    ├── ...
    └── test/          # 独立测试图片，不参与训练/建 bank
```

例如：

```text
data/phone/
├── good/
├── shao1/
├── shao2/
├── shao3/
└── test/
```

注意：

- `good/` 只能放正常样本。
- 已知缺陷目录名称就是类别名称。
- `test/` 不用于 PatchCore 训练，也不用于 DINO bank 建立。
- 生产 PASS/NG 阈值应使用独立校准集，不要用最终 test 集反推。
