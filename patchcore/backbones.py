"""骨干网络注册模块。

本模块维护了一个预训练骨干网络的注册表，支持 30+ 种网络架构，
通过统一的 load() 接口按名称加载预训练模型。

支持的骨干网络来源：
- torchvision.models: ResNet, VGG, AlexNet, WideResNet 等
- timm (PyTorch Image Models): ViT, Swin, EfficientNet, DenseNet, ResNeSt 等
- pretrainedmodels: BN-Inception
"""

import timm  # noqa: PyTorch Image Models - 提供大量预训练模型
import torchvision.models as models  # noqa: torchvision 预训练模型

# 骨干网络注册表
# 键为模型名称，值为用于动态加载模型的 Python 表达式字符串
_BACKBONES = {
    # --- torchvision 经典模型 ---
    "alexnet": "models.alexnet(pretrained=True)",
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    # --- ResNet 系列 ---
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    # --- BN-Inception（来自 pretrainedmodels 库）---
    "bninception": 'pretrainedmodels.__dict__["bninception"]'
    '(pretrained="imagenet", num_classes=1000)',
    # --- timm 模型 ---
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    # ResNetV2 (Big Transfer) 系列
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    # MNASNet 系列
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    # DenseNet 系列
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    # Inception V4
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    # Vision Transformer (ViT) 系列
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    # DeiT (Data-efficient Image Transformers)
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    # Swin Transformer
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    # EfficientNet 系列
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
}


def load(name):
    """按名称加载预训练骨干网络。

    通过 eval() 动态执行注册表中的表达式字符串来加载模型。
    加载的模型已包含预训练权重（ImageNet 预训练）。

    Args:
        name: 骨干网络名称，必须是 _BACKBONES 中的键
              如 'wideresnet50', 'resnet101', 'densenet201' 等

    Returns:
        加载的预训练模型实例

    Raises:
        KeyError: 如果 name 不在 _BACKBONES 中

    Example:
        >>> model = load("wideresnet50")
        >>> model = load("vit_base")
    """
    return eval(_BACKBONES[name])