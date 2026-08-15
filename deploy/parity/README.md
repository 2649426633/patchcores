# Python ↔ ONNX/C# parity workspace

此目录用于放正式上线前的逐级一致性验证工具。

目标不是“能跑”，而是确认原 Python 与 ONNX/C# 在同一输入上尽可能 1:1 对齐。

推荐后续文件：

```text
deploy/parity/
├── export_python_reference.py   # 原 Python 导出固定 tile 的中间结果
├── compare_reference.py         # 比较 Python / C# 输出
├── samples/                     # 固定验收图片/路径说明，不提交大图可只留清单
└── README.md
```

必须比较：

```text
1. normalized input tensor
2. Patch embedding [1600,1024]
3. nearest-neighbour patch score [1600]
4. anomaly map
5. bbox
6. DINO CLS [384]
7. DINO Center [384]
8. similarity / final class
```

建议验收规则：

- 第 2 步不通过：停止后续调参，先修 PatchCore ONNX feature graph。
- 第 2 步通过、第 3 步不通过：检查 squared-L2、Memory Bank 内容和行顺序。
- 第 3 步通过但 map/bbox 不通过：检查 resize、bilinear、Gaussian、threshold、connected components、tile merge。
- DINO embedding 不通过：检查 RGB/NCHW/ImageNet normalization 与 center-patch pooling。
- embedding 都通过但分类不同：检查 labels、bank 行顺序、cosine 和 0.5/0.5 融合。

当前目录只是固定了验收位置；正式 parity 脚本仍需要实现并在现有 phone 数据上跑通。
