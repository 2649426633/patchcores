# Converted deployment product packages

此目录只放由原 Python 模型转换出来的 C# / WinForms 产品包。

禁止在 C# 现场重新随机建立 PatchCore Memory 或 DINO defect bank。

推荐结构：

```text
deploy/products/
└── <product>/
    ├── patchcore_memory.bin
    ├── defect_cls.bin
    ├── defect_center.bin
    ├── product_model.json
    ├── conversion_report.json
    └── support_rois/        # 可选，人工复核用
```

产品包由下面脚本生成：

```powershell
python deploy\convert_python_product.py ^
  --product <product> ^
  --patchcore-model-dir industrial_anomaly\products\<product>\models\patchcore ^
  --defect-bank-dir industrial_anomaly\products\<product>\models\defect_bank ^
  --output-dir deploy\products\<product>
```

正式产品必须满足：

```text
ProductModelSource = python_export
PatchCoreMemoryStrategy = python_faiss_memory_exact
```

`patchcore_memory.bin` 来自原 Python FAISS coreset memory；`defect_cls.bin` 和 `defect_center.bin` 直接来自原 Python DINO exemplar bank。
