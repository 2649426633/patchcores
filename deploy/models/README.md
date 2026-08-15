# Generated ONNX engines

此目录放本地导出的通用 ONNX 引擎，不提交 `.onnx` 文件到 GitHub。

运行：

```powershell
python deploy\export_onnx.py --model patchcore --device cpu --exporter legacy --skip-verify
python deploy\export_onnx.py --model dinov2 --device cpu --exporter legacy --skip-verify
```

生成：

```text
deploy/models/
├── patchcore_feature.onnx
├── dinov2_feature.onnx
└── engine_config.json
```

说明：

- ONNX 是通用特征引擎，不包含具体产品的正常 Memory 或缺陷类别。
- 产品差异由 `deploy/products/<product>/` 下的三个 `.bin` 与 `product_model.json` 提供。
- 正式发布前必须完成 Python ↔ ONNX feature parity 验证。
