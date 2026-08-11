"""PatchCore 异常检测方法实现。

本文档实现了 PatchCore 算法的核心逻辑，包括：
- PatchCore: 主模型类，封装训练（记忆库构建）和推理（异常检测）全流程
- PatchMaker: 负责将特征图切分为 patch 特征，以及反向重组操作

参考文献: Roth et al. (2021), Towards Total Recall in Industrial Anomaly Detection
"""
import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import tqdm

import patchcore
import patchcore.backbones
import patchcore.common
import patchcore.sampler

LOGGER = logging.getLogger(__name__)


class PatchCore(torch.nn.Module):
    """PatchCore 异常检测核心类。

    该类的设计思想是：使用预训练的骨干网络提取图像局部 patch 特征，
    通过贪婪核心集（Coreset）子采样构建紧凑的记忆库，最后利用最近邻搜索
    进行异常评分与分割。

    主要流程:
        训练阶段: fit() -> _fill_memory_bank() -> 特征提取 -> coreset采样 -> FAISS索引
        推理阶段: predict() -> _predict() -> 特征提取 -> 最近邻搜索 -> 分数计算 -> 分割图
    """

    def __init__(self, device):
        """初始化 PatchCore 实例。

        Args:
            device: 运行设备（'cpu' 或 'cuda:0' 等）
        """
        super(PatchCore, self).__init__()
        self.device = device

    def load(
        self,
        backbone,
        layers_to_extract_from,
        device,
        input_shape,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize=3,
        patchstride=1,
        anomaly_score_num_nn=1,
        featuresampler=patchcore.sampler.IdentitySampler(),
        nn_method=patchcore.common.FaissNN(False, 4),
        **kwargs,
    ):
        """加载并配置 PatchCore 模型的所有子模块。

        Args:
            backbone: 预训练骨干网络（如 wideresnet50）
            layers_to_extract_from: 需要提取特征的网络层列表（如 ['layer2', 'layer3']）
            device: 运行设备
            input_shape: 输入图像形状 [C, H, W]
            pretrain_embed_dimension: 预训练嵌入维度（将各层特征映射到的中间维度）
            target_embed_dimension: 目标嵌入维度（最终聚合后的特征维度）
            patchsize: patch 的局部邻域大小（默认 3）
            patchstride: patch 滑动步长（默认 1）
            anomaly_score_num_nn: 用于异常评分的最近邻数量（默认 1）
            featuresampler: 特征子采样器实例（默认 IdentitySampler）
            nn_method: 最近邻搜索方法实例（默认 FaissNN）
        """
        # 将骨干网络移动到指定设备
        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        # 初始化 patch 切分器
        self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        # 前向传播子模块字典
        self.forward_modules = torch.nn.ModuleDict({})

        # 1) 特征聚合器：通过 forward hooks 从 backbone 指定层提取特征
        feature_aggregator = patchcore.common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        # 2) 预处理模块：将各层特征映射到统一维度
        preprocessing = patchcore.common.Preprocessing(
            feature_dimensions, pretrain_embed_dimension
        )
        self.forward_modules["preprocessing"] = preprocessing

        # 3) 聚合模块：将多层特征聚合到目标维度
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = patchcore.common.Aggregator(
            target_dim=target_embed_dimension
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        # 4) 最近邻异常评分器
        self.anomaly_scorer = patchcore.common.NearestNeighbourScorer(
            n_nearest_neighbours=anomaly_score_num_nn, nn_method=nn_method
        )

        # 5) 分割图生成器：将 patch 级分数上采样到原图分辨率
        self.anomaly_segmentor = patchcore.common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

        # 特征子采样器（用于 coreset 降采样）
        self.featuresampler = featuresampler

    def embed(self, data):
        """将输入数据转换为特征嵌入。

        Args:
            data: 可以是单个图像 Tensor 或 DataLoader 对象

        Returns:
            特征嵌入列表或单个特征嵌入
        """
        if isinstance(data, torch.utils.data.DataLoader):
            features = []
            for image in data:
                if isinstance(image, dict):
                    image = image["image"]
                with torch.no_grad():
                    input_image = image.to(torch.float).to(self.device)
                    features.append(self._embed(input_image))
            return features
        return self._embed(data)

    def _embed(self, images, detach=True, provide_patch_shapes=False):
        """提取图像的特征嵌入。

        完整流程:
            1. 通过 backbone 提取各层特征图
            2. 将特征图切分为 patch 特征
            3. 将不同层级的 patch 特征对齐到相同空间分辨率
            4. 对特征进行维度压缩和聚合

        Args:
            images: 输入图像 Tensor [B, C, H, W]
            detach: 是否从计算图中分离（默认 True，避免梯度累积）
            provide_patch_shapes: 是否返回各层 patch 的空间形状信息

        Returns:
            detach=True: numpy 数组形式的特征嵌入
            detach=False: torch Tensor 形式的特征嵌入
        """

        def _detach(features):
            """将特征从计算图分离并转为 numpy 数组（如果需要）。"""
            if detach:
                return [x.detach().cpu().numpy() for x in features]
            return features

        # 切换到评估模式并提取特征
        _ = self.forward_modules["feature_aggregator"].eval()
        with torch.no_grad():
            features = self.forward_modules["feature_aggregator"](images)

        # 提取指定层的特征图
        features = [features[layer] for layer in self.layers_to_extract_from]

        # 将各层特征图切分为 patch，并保留空间信息
        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]  # 各层 patch 的空间尺寸
        features = [x[0] for x in features]  # 各层 patch 特征
        ref_num_patches = patch_shapes[0]  # 以第一层为参考

        # 将各层 patch 特征对齐到相同的空间分辨率
        # 不同 backbone 层的特征图可能具有不同的空间尺寸，
        # 这里使用双线性插值将它们统一到参考层的分辨率
        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            # 将 patch 特征重组为空间特征图格式
            # [B*N, C, P, P] -> [B, H_p, W_p, C, P, P]
            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            # 将通道维度移到前面: [B, H_p, W_p, C, P, P] -> [B, C, P, P, H_p, W_p]
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            # 合并 batch 和 patch 维度进行插值: [B*C*P*P, H_p, W_p]
            _features = _features.reshape(-1, *_features.shape[-2:])
            # 双线性插值到参考分辨率
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            # 恢复原始形状: [B, C, P, P, H_ref, W_ref]
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            # 恢复空间维度: [B, H_ref, W_ref, C, P, P]
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            # 恢复为 patch 格式: [B*H_ref*W_ref, C, P, P]
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features

        # 合并 batch 和空间维度: [B*N, C, P, P]
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # 预处理：将各层特征映射到统一维度
        features = self.forward_modules["preprocessing"](features)
        # 聚合：将多层特征融合到目标维度
        features = self.forward_modules["preadapt_aggregator"](features)

        if provide_patch_shapes:
            return _detach(features), patch_shapes
        return _detach(features)

    def fit(self, training_data):
        """训练 PatchCore 模型。

        对训练数据（仅包含正常样本）提取特征嵌入，经过 coreset 子采样
        后填充到记忆库（FAISS 索引）中。

        Args:
            training_data: 训练数据 DataLoader
        """
        self._fill_memory_bank(training_data)

    def _fill_memory_bank(self, input_data):
        """构建记忆库。

        遍历所有训练数据，提取特征嵌入，经过 coreset 子采样后
        构建 FAISS 最近邻搜索索引。

        Args:
            input_data: 训练数据 DataLoader
        """
        _ = self.forward_modules.eval()

        def _image_to_features(input_image):
            """将单张图像转换为特征嵌入。"""
            with torch.no_grad():
                input_image = input_image.to(torch.float).to(self.device)
                return self._embed(input_image)

        features = []
        with tqdm.tqdm(
            input_data, desc="Computing support features...", position=1, leave=False
        ) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    image = image["image"]
                features.append(_image_to_features(image))

        # 合并所有训练特征的 patch 维度: [N_total, D]
        features = np.concatenate(features, axis=0)
        # 通过 coreset 采样减少记忆库大小
        features = self.featuresampler.run(features)

        # 构建最近邻索引（记忆库）
        self.anomaly_scorer.fit(detection_features=[features])

    def predict(self, data):
        """对输入数据进行异常检测。

        Args:
            data: 可以是单个图像 Tensor batch 或 DataLoader 对象

        Returns:
            对于 DataLoader: (scores, masks, labels_gt, masks_gt)
            对于单 batch: (scores, masks)
        """
        if isinstance(data, torch.utils.data.DataLoader):
            return self._predict_dataloader(data)
        return self._predict(data)

    def _predict_dataloader(self, dataloader):
        """对完整 DataLoader 进行异常检测。

        Args:
            dataloader: 测试数据 DataLoader

        Returns:
            scores: 图像级异常分数列表
            masks: 像素级异常分割掩码列表
            labels_gt: 真实标签列表
            masks_gt: 真实掩码列表
        """
        _ = self.forward_modules.eval()

        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        with tqdm.tqdm(dataloader, desc="Inferring...", leave=False) as data_iterator:
            for image in data_iterator:
                if isinstance(image, dict):
                    labels_gt.extend(image["is_anomaly"].numpy().tolist())
                    masks_gt.extend(image["mask"].numpy().tolist())
                    image = image["image"]
                _scores, _masks = self._predict(image)
                for score, mask in zip(_scores, _masks):
                    scores.append(score)
                    masks.append(mask)
        return scores, masks, labels_gt, masks_gt

    def _predict(self, images):
        """对一批图像进行异常检测。

        Args:
            images: 输入图像 Tensor [B, C, H, W]

        Returns:
            image_scores: 图像级异常分数列表
            masks: 像素级异常分割掩码列表
        """
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        with torch.no_grad():
            # 提取特征嵌入
            features, patch_shapes = self._embed(images, provide_patch_shapes=True)
            features = np.asarray(features)

            # 通过最近邻搜索获取每个 patch 的异常分数
            # patch_scores: 每个 patch 到最近邻的平均距离
            patch_scores = image_scores = self.anomaly_scorer.predict([features])[0]

            # 将 patch 分数重组为图像级分数
            image_scores = self.patch_maker.unpatch_scores(
                image_scores, batchsize=batchsize
            )
            image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
            # 沿空间维度取 max，得到图像级异常分数
            image_scores = self.patch_maker.score(image_scores)

            # 将 patch 分数重组为空间特征图
            patch_scores = self.patch_maker.unpatch_scores(
                patch_scores, batchsize=batchsize
            )
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])

            # 上采样到原图分辨率并应用高斯平滑，生成分割掩码
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

        return [score for score in image_scores], [mask for mask in masks]

    @staticmethod
    def _params_file(filepath, prepend=""):
        """获取参数文件的完整路径。

        Args:
            filepath: 模型保存目录
            prepend: 文件名前缀（用于 Ensemble 模型区分）

        Returns:
            参数文件的完整路径
        """
        return os.path.join(filepath, prepend + "patchcore_params.pkl")

    def save_to_path(self, save_path: str, prepend: str = "") -> None:
        """将 PatchCore 模型保存到指定路径。

        保存内容包括:
            - FAISS 最近邻搜索索引
            - 模型参数字典（pickle 序列化）

        Args:
            save_path: 保存目录路径
            prepend: 文件名前缀（用于 Ensemble 模型区分）
        """
        LOGGER.info("Saving PatchCore data.")
        self.anomaly_scorer.save(
            save_path, save_features_separately=False, prepend=prepend
        )
        patchcore_params = {
            "backbone.name": self.backbone.name,
            "layers_to_extract_from": self.layers_to_extract_from,
            "input_shape": self.input_shape,
            "pretrain_embed_dimension": self.forward_modules[
                "preprocessing"
            ].output_dim,
            "target_embed_dimension": self.forward_modules[
                "preadapt_aggregator"
            ].target_dim,
            "patchsize": self.patch_maker.patchsize,
            "patchstride": self.patch_maker.stride,
            "anomaly_scorer_num_nn": self.anomaly_scorer.n_nearest_neighbours,
        }
        with open(self._params_file(save_path, prepend), "wb") as save_file:
            pickle.dump(patchcore_params, save_file, pickle.HIGHEST_PROTOCOL)

    def load_from_path(
        self,
        load_path: str,
        device: torch.device,
        nn_method: patchcore.common.FaissNN(False, 4),
        prepend: str = "",
    ) -> None:
        """从指定路径加载预训练 PatchCore 模型。

        加载流程:
            1. 读取参数文件恢复模型配置
            2. 根据配置重建 backbone 网络
            3. 加载 FAISS 搜索索引

        Args:
            load_path: 模型加载目录路径
            device: 运行设备
            nn_method: 最近邻搜索方法实例
            prepend: 文件名前缀（用于 Ensemble 模型区分）
        """
        LOGGER.info("Loading and initializing PatchCore.")
        with open(self._params_file(load_path, prepend), "rb") as load_file:
            patchcore_params = pickle.load(load_file)
        # 从配置文件恢复 backbone 实例
        patchcore_params["backbone"] = patchcore.backbones.load(
            patchcore_params["backbone.name"]
        )
        patchcore_params["backbone"].name = patchcore_params["backbone.name"]
        del patchcore_params["backbone.name"]
        self.load(**patchcore_params, device=device, nn_method=nn_method)

        self.anomaly_scorer.load(load_path, prepend)


# --- 图像处理辅助类 ---


class PatchMaker:
    """Patch 切分与重组工具类。

    负责将特征图切分为局部 patch 特征，以及将 patch 级分数重组为图像级分数。
    核心操作使用 torch.nn.Unfold 实现滑动窗口切分。
    """

    def __init__(self, patchsize, stride=None):
        """初始化 PatchMaker。

        Args:
            patchsize: patch 的局部邻域大小（如 3 表示 3x3 邻域）
            stride: 滑动步长（默认等于 patchsize，即无重叠）
        """
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        """将特征图张量切分为 patch 特征。

        使用 torch.nn.Unfold 实现滑窗操作，每个空间位置取 patchsize x patchsize 的邻域。

        Args:
            features: 输入特征图 [B, C, W, H]
            return_spatial_info: 是否返回空间形状信息

        Returns:
            return_spatial_info=False: patch 特征 [B*W*H, C, patchsize, patchsize]
            return_spatial_info=True: (patch特征, [W_patches, H_patches])
        """
        # 计算填充量，使 patch 中心对齐
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)

        # 计算各维度的 patch 数量
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))

        # 重组为 [B, N, C, patchsize, patchsize]
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        """将 patch 分数重组为按 batch 维度的格式。

        Args:
            x: patch 分数 [B*N, ...]
            batchsize: 批次大小

        Returns:
            重组后的分数 [B, N, ...]
        """
        return x.reshape(batchsize, -1, *x.shape[1:])

    def score(self, x):
        """沿空间维度递归取最大值，得到图像级异常分数。

        对于多维张量，逐层取 max 直到只剩 batch 维度。

        Args:
            x: 输入张量（支持 numpy 或 torch）

        Returns:
            图像级异常分数 [B] 或 [B, 1]
        """
        was_numpy = False
        if isinstance(x, np.ndarray):
            was_numpy = True
            x = torch.from_numpy(x)
        # 递归沿最后一维取 max，得到每个 batch 的最大异常分数
        while x.ndim > 1:
            x = torch.max(x, dim=-1).values
        if was_numpy:
            return x.numpy()
        return x