"""PatchCore 通用组件模块。

本模块提供了 PatchCore 算法必需的底层组件，包括：
- FAISS 最近邻搜索（精确 & 近似）
- 特征合并策略（平均 & 拼接）
- 特征预处理与聚合
- 网络特征提取器（基于 forward hooks）
- 最近邻异常评分器
- 分割图生成器（上采样 + 高斯平滑）
"""
import copy
import os
import pickle
from typing import List
from typing import Union

import faiss
import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F


# ============================================================================
# FAISS 最近邻搜索
# ============================================================================


class FaissNN(object):
    """基于 FAISS 的精确最近邻搜索封装。

    使用 IndexFlatL2 索引实现精确的 L2 距离最近邻搜索。
    支持 GPU 加速和 CPU 回退。
    """

    def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
        """初始化 FAISS 最近邻搜索器。

        Args:
            on_gpu: 是否在 GPU 上执行搜索（需要 faiss-gpu 包）
            num_workers: FAISS 使用的线程数
        """
        faiss.omp_set_num_threads(num_workers)
        self.on_gpu = on_gpu
        self.search_index = None

    def _gpu_cloner_options(self):
        """获取 GPU 克隆选项（子类可重写以自定义 GPU 配置）。"""
        return faiss.GpuClonerOptions()

    def _index_to_gpu(self, index):
        """将索引从 CPU 移动到 GPU。

        Args:
            index: FAISS CPU 索引

        Returns:
            GPU 索引（如果 on_gpu=True），否则返回原 CPU 索引
        """
        if self.on_gpu:
            return faiss.index_cpu_to_gpu(
                faiss.StandardGpuResources(), 0, index, self._gpu_cloner_options()
            )
        return index

    def _index_to_cpu(self, index):
        """将索引从 GPU 移动到 CPU。

        Args:
            index: FAISS GPU 索引

        Returns:
            CPU 索引（如果 on_gpu=True），否则返回原索引
        """
        if self.on_gpu:
            return faiss.index_gpu_to_cpu(index)
        return index

    def _create_index(self, dimension):
        """创建新的 FAISS 索引。

        Args:
            dimension: 特征向量的维度

        Returns:
            新创建的 FAISS 索引（CPU 或 GPU 版本）
        """
        if self.on_gpu:
            return faiss.GpuIndexFlatL2(
                faiss.StandardGpuResources(), dimension, faiss.GpuIndexFlatConfig()
            )
        return faiss.IndexFlatL2(dimension)

    def fit(self, features: np.ndarray) -> None:
        """将特征添加到 FAISS 搜索索引中。

        Args:
            features: 特征数组 [N, D]，N 为样本数，D 为特征维度
        """
        if self.search_index:
            self.reset_index()
        self.search_index = self._create_index(features.shape[-1])
        self._train(self.search_index, features)
        self.search_index.add(features)

    def _train(self, _index, _features):
        """训练索引（对于 IndexFlatL2 无需训练，子类可重写）。"""
        pass

    def run(
        self,
        n_nearest_neighbours,
        query_features: np.ndarray,
        index_features: np.ndarray = None,
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """执行最近邻搜索。

        Args:
            n_nearest_neighbours: 需要返回的最近邻数量
            query_features: 查询特征 [N_q, D]
            index_features: 索引特征 [N_i, D]（可选，若提供则临时构建索引）

        Returns:
            (distances, indices): 距离和最近邻索引
        """
        if index_features is None:
            return self.search_index.search(query_features, n_nearest_neighbours)

        # 当提供了 index_features 时，临时构建一个搜索索引
        search_index = self._create_index(index_features.shape[-1])
        self._train(search_index, index_features)
        search_index.add(index_features)
        return search_index.search(query_features, n_nearest_neighbours)

    def save(self, filename: str) -> None:
        """将索引保存到文件。

        Args:
            filename: 保存路径
        """
        faiss.write_index(self._index_to_cpu(self.search_index), filename)

    def load(self, filename: str) -> None:
        """从文件加载索引。

        Args:
            filename: 索引文件路径
        """
        self.search_index = self._index_to_gpu(faiss.read_index(filename))

    def reset_index(self):
        """重置搜索索引，释放内存。"""
        if self.search_index:
            self.search_index.reset()
            self.search_index = None


class ApproximateFaissNN(FaissNN):
    """基于 FAISS 的近似最近邻搜索封装。

    使用 IndexIVFPQ（倒排文件 + 乘积量化）实现近似搜索，
    在 GPU 上使用 float16 精度以进一步加速和减少内存占用。
    相比精确搜索，近似搜索速度更快，但可能略微降低精度。
    """

    def _train(self, index, features):
        """训练 IVF 索引（IVF 索引需要训练步骤）。

        Args:
            index: IVF 索引
            features: 训练特征
        """
        index.train(features)

    def _gpu_cloner_options(self):
        """获取 GPU 克隆选项，使用 float16 精度。

        Returns:
            配置了 float16 的 GpuClonerOptions
        """
        cloner = faiss.GpuClonerOptions()
        cloner.useFloat16 = True
        return cloner

    def _create_index(self, dimension):
        """创建 IVF+PQ 近似搜索索引。

        参数说明:
            - n_centroids=512: 聚类中心数
            - sub-quantizers=64: 乘积量化的子量化器数量
            - nbits=8: 每个子量化器的编码位数

        Args:
            dimension: 特征向量维度

        Returns:
            IndexIVFPQ 索引
        """
        index = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(dimension),
            dimension,
            512,  # n_centroids: 聚类中心数
            64,  # sub-quantizers: 子量化器数量
            8,  # nbits per code: 每个编码的位数
        )
        return self._index_to_gpu(index)


# ============================================================================
# 特征合并策略
# ============================================================================


class _BaseMerger:
    """特征合并基类。

    负责将多个层级的特征嵌入合并为统一的特征表示。
    """

    def __init__(self):
        """初始化特征合并器。"""

    def merge(self, features: list):
        """合并特征列表。

        Args:
            features: 特征列表，每个元素为 [N, C, H, W] 或 [N, C]

        Returns:
            合并后的特征 [N, total_dim]
        """
        features = [self._reduce(feature) for feature in features]
        return np.concatenate(features, axis=1)


class AverageMerger(_BaseMerger):
    """平均合并策略。

    将每个特征图沿空间维度取平均，压缩为 [N, C] 后拼接。
    """

    @staticmethod
    def _reduce(features):
        """将特征图沿空间维度平均池化。

        Args:
            features: 特征图 [N, C, W, H]

        Returns:
            平均池化后的特征 [N, C]
        """
        return features.reshape([features.shape[0], features.shape[1], -1]).mean(
            axis=-1
        )


class ConcatMerger(_BaseMerger):
    """拼接合并策略。

    将每个特征图直接展平为 [N, C*W*H] 后拼接。
    相比 AverageMerger 保留了更多空间信息，但维度更高。
    """

    @staticmethod
    def _reduce(features):
        """将特征图展平为向量。

        Args:
            features: 特征图 [N, C, W, H]

        Returns:
            展平后的特征 [N, C*W*H]
        """
        return features.reshape(len(features), -1)


# ============================================================================
# 特征预处理与聚合
# ============================================================================


class Preprocessing(torch.nn.Module):
    """特征预处理模块。

    将各 backbone 层提取的特征映射到统一的中间维度。
    每个输入层使用独立的 MeanMapper 模块进行维度变换。
    """

    def __init__(self, input_dims, output_dim):
        """初始化预处理模块。

        Args:
            input_dims: 各层输入特征的通道维度列表 [C1, C2, ...]
            output_dim: 目标输出维度
        """
        super(Preprocessing, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 为每个输入层创建独立的维度映射器
        self.preprocessing_modules = torch.nn.ModuleList()
        for input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        """前向传播：对各层特征分别进行维度映射。

        Args:
            features: 各层特征列表 [feat_layer1, feat_layer2, ...]

        Returns:
            堆叠后的特征 [B, n_layers, output_dim]
        """
        _features = []
        for module, feature in zip(self.preprocessing_modules, features):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    """自适应平均池化维度映射器。

    通过 adaptive_avg_pool1d 将任意长度的特征向量压缩到指定维度。
    """

    def __init__(self, preprocessing_dim):
        """初始化 MeanMapper。

        Args:
            preprocessing_dim: 目标输出维度
        """
        super(MeanMapper, self).__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        """前向传播：自适应平均池化到目标维度。

        Args:
            features: 输入特征 [B, N, D] 或 [B*N, D]

        Returns:
            压缩后的特征 [B*N, preprocessing_dim]
        """
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    """特征聚合模块。

    将多层特征堆叠后，通过自适应平均池化聚合到目标维度。
    输入 [B, n_layers, D] -> 输出 [B, target_dim]
    """

    def __init__(self, target_dim):
        """初始化聚合器。

        Args:
            target_dim: 目标输出维度
        """
        super(Aggregator, self).__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """前向传播：将多层特征聚合到目标维度。

        Args:
            features: 堆叠的多层特征 [B, n_layers, D]

        Returns:
            聚合后的特征 [B, target_dim]
        """
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


# ============================================================================
# 分割图生成器
# ============================================================================


class RescaleSegmentor:
    """分割图生成器。

    将 patch 级异常分数上采样到原图分辨率，并应用高斯平滑，
    生成最终的像素级异常分割掩码。
    """

    def __init__(self, device, target_size=224):
        """初始化分割图生成器。

        Args:
            device: 运行设备
            target_size: 目标图像尺寸 (H, W)
        """
        self.device = device
        self.target_size = target_size
        self.smoothing = 4  # 高斯平滑的 sigma 值

    def convert_to_segmentation(self, patch_scores):
        """将 patch 分数转换为分割掩码。

        流程:
            1. 双线性插值上采样到目标尺寸
            2. 高斯滤波平滑

        Args:
            patch_scores: patch 级异常分数 [B, H_p, W_p]

        Returns:
            分割掩码列表，每个元素为 [H, W]
        """
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)  # [B, 1, H_p, W_p]
            # 双线性插值上采样
            _scores = F.interpolate(
                _scores, size=self.target_size, mode="bilinear", align_corners=False
            )
            _scores = _scores.squeeze(1)
            patch_scores = _scores.cpu().numpy()

        # 对每个样本应用高斯滤波平滑
        return [
            ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
            for patch_score in patch_scores
        ]


# ============================================================================
# 网络特征提取器
# ============================================================================


class NetworkFeatureAggregator(torch.nn.Module):
    """基于 Forward Hook 的网络特征提取器。

    通过在 backbone 指定层注册 forward hook，高效地提取中间层特征。
    到达最后一层后通过抛出异常提前终止计算，避免不必要的计算开销。
    """

    def __init__(self, backbone, layers_to_extract_from, device):
        """初始化特征提取器。

        Args:
            backbone: 预训练骨干网络
            layers_to_extract_from: 需要提取特征的层名列表（如 ['layer2', 'layer3']）
            device: 运行设备
        """
        super(NetworkFeatureAggregator, self).__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        self.device = device

        # 清理旧的 hook 句柄，避免重复注册
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        # 为每个目标层注册 forward hook
        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(
                self.outputs, extract_layer, layers_to_extract_from[-1]
            )
            # 支持点号路径访问（如 '0.layer2' 或 'features.denseblock2'）
            if "." in extract_layer:
                extract_block, extract_idx = extract_layer.split(".")
                network_layer = backbone.__dict__["_modules"][extract_block]
                if extract_idx.isnumeric():
                    extract_idx = int(extract_idx)
                    network_layer = network_layer[extract_idx]
                else:
                    network_layer = network_layer.__dict__["_modules"][extract_idx]
            else:
                network_layer = backbone.__dict__["_modules"][extract_layer]

            # 注册 hook：对于 Sequential 层，hook 其最后一个子层
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(
                    network_layer.register_forward_hook(forward_hook)
                )
        self.to(self.device)

    def forward(self, images):
        """前向传播：提取中间层特征。

        通过捕获 LastLayerToExtractReachedException 提前终止前向传播，
        避免计算不需要的更深层特征。

        Args:
            images: 输入图像 [B, C, H, W]

        Returns:
            各层特征输出的字典 {layer_name: feature_tensor}
        """
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs

    def feature_dimensions(self, input_shape):
        """计算各目标层输出特征的通道维度。

        Args:
            input_shape: 输入图像形状 [C, H, W]

        Returns:
            各层通道维度列表 [C1, C2, ...]
        """
        _input = torch.ones([1] + list(input_shape)).to(self.device)
        _output = self(_input)
        return [_output[layer].shape[1] for layer in self.layers_to_extract_from]


class ForwardHook:
    """Forward Hook 回调类。

    注册到 backbone 层，在每次前向传播时捕获该层的输出特征。
    当到达需要提取的最后一层时，抛出异常提前终止前向传播。
    """

    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        """初始化 Forward Hook。

        Args:
            hook_dict: 存储特征输出的字典
            layer_name: 当前层的名称
            last_layer_to_extract: 需要提取的最后一层名称
        """
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        # 标记当前层是否为需要提取的最后一层
        self.raise_exception_to_break = copy.deepcopy(
            layer_name == last_layer_to_extract
        )

    def __call__(self, module, input, output):
        """Hook 回调函数。

        Args:
            module: 被 hook 的模块
            input: 模块的输入
            output: 模块的输出（存储到 hook_dict 中）
        """
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class LastLayerToExtractReachedException(Exception):
    """到达最后一层时抛出的异常，用于提前终止前向传播。"""
    pass


# ============================================================================
# 最近邻异常评分器
# ============================================================================


class NearestNeighbourScorer(object):
    """最近邻异常评分器。

    通过查询特征与记忆库中最近邻的距离来评估异常程度。
    距离越大表示越异常。
    """

    def __init__(self, n_nearest_neighbours: int, nn_method=FaissNN(False, 4)) -> None:
        """初始化最近邻异常评分器。

        Args:
            n_nearest_neighbours: 用于评分的最近邻数量
            nn_method: 最近邻搜索方法实例
        """
        self.feature_merger = ConcatMerger()

        self.n_nearest_neighbours = n_nearest_neighbours
        self.nn_method = nn_method

        # 图像级最近邻搜索（用于获取异常分数）
        self.imagelevel_nn = lambda query: self.nn_method.run(
            n_nearest_neighbours, query
        )
        # 像素级最近邻搜索（用于获取每个 patch 的最近邻）
        self.pixelwise_nn = lambda query, index: self.nn_method.run(1, query, index)

    def fit(self, detection_features: List[np.ndarray]) -> None:
        """构建最近邻搜索索引。

        Args:
            detection_features: 训练特征列表
                [[N_i x D] for i in n] 包含所有训练图像的特征向量
        """
        self.detection_features = self.feature_merger.merge(
            detection_features,
        )
        self.nn_method.fit(self.detection_features)

    def predict(
        self, query_features: List[np.ndarray]
    ) -> Union[np.ndarray, np.ndarray, np.ndarray]:
        """对查询特征进行异常评分。

        在记忆库中搜索每个查询特征向量的最近邻，
        返回平均距离作为异常分数。

        Args:
            query_features: 查询特征列表

        Returns:
            (anomaly_scores, query_distances, query_nns):
                anomaly_scores: 平均最近邻距离 [N]
                query_distances: 所有最近邻的距离
                query_nns: 最近邻的索引
        """
        query_features = self.feature_merger.merge(
            query_features,
        )
        query_distances, query_nns = self.imagelevel_nn(query_features)
        # 异常分数 = 到最近邻的平均距离
        anomaly_scores = np.mean(query_distances, axis=-1)
        return anomaly_scores, query_distances, query_nns

    # --- 文件操作辅助方法 ---

    @staticmethod
    def _detection_file(folder, prepend=""):
        """获取特征文件的路径。"""
        return os.path.join(folder, prepend + "nnscorer_features.pkl")

    @staticmethod
    def _index_file(folder, prepend=""):
        """获取索引文件的路径。"""
        return os.path.join(folder, prepend + "nnscorer_search_index.faiss")

    @staticmethod
    def _save(filename, features):
        """保存特征到文件。"""
        if features is None:
            return
        with open(filename, "wb") as save_file:
            pickle.dump(features, save_file, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def _load(filename: str):
        """从文件加载特征。"""
        with open(filename, "rb") as load_file:
            return pickle.load(load_file)

    def save(
        self,
        save_folder: str,
        save_features_separately: bool = False,
        prepend: str = "",
    ) -> None:
        """保存评分器状态。

        Args:
            save_folder: 保存目录
            save_features_separately: 是否单独保存特征文件
            prepend: 文件名前缀（用于 Ensemble 模型区分）
        """
        self.nn_method.save(self._index_file(save_folder, prepend))
        if save_features_separately:
            self._save(
                self._detection_file(save_folder, prepend), self.detection_features
            )

    def save_and_reset(self, save_folder: str) -> None:
        """保存并重置索引（释放内存）。"""
        self.save(save_folder)
        self.nn_method.reset_index()

    def load(self, load_folder: str, prepend: str = "") -> None:
        """加载评分器状态。

        Args:
            load_folder: 加载目录
            prepend: 文件名前缀
        """
        self.nn_method.load(self._index_file(load_folder, prepend))
        if os.path.exists(self._detection_file(load_folder, prepend)):
            self.detection_features = self._load(
                self._detection_file(load_folder, prepend)
            )