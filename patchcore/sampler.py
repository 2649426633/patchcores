"""特征采样器模块。

本模块实现了多种特征子采样策略，用于从海量训练 patch 特征中
选取最具代表性的子集，从而构建紧凑的记忆库。

采样器层次结构:
    BaseSampler (ABC)
    ├── IdentitySampler      # 不采样，直接返回
    ├── RandomSampler        # 随机均匀采样
    ├── GreedyCoresetSampler # 精确贪婪核心集采样
    └── ApproximateGreedyCoresetSampler  # 近似贪婪核心集采样（内存友好）

核心集采样的目标是从 N 个特征中选出 K 个最具代表性的子集，
使得子集到全集的最大距离最小化（minimax facility location）。
"""
import abc
from typing import Union

import numpy as np
import torch
import tqdm


class IdentitySampler:
    """恒等采样器。

    不做任何采样，直接返回原始特征。用于不使用采样的场景。
    """

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """直接返回原始特征。

        Args:
            features: 输入特征 [N, D]

        Returns:
            原始特征（不做任何修改）
        """
        return features


class BaseSampler(abc.ABC):
    """特征采样基类。

    定义了采样器的基本接口和类型保持机制。
    所有采样器都需要继承此类并实现 run() 方法。
    """

    def __init__(self, percentage: float):
        """初始化采样器。

        Args:
            percentage: 采样比例，必须在 (0, 1) 之间

        Raises:
            ValueError: 如果 percentage 不在 (0, 1) 范围内
        """
        if not 0 < percentage < 1:
            raise ValueError("Percentage value not in (0, 1).")
        self.percentage = percentage

    @abc.abstractmethod
    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """执行采样（抽象方法，子类必须实现）。

        Args:
            features: 输入特征 [N, D]

        Returns:
            采样后的特征 [K, D]，其中 K = N * percentage
        """
        pass

    def _store_type(self, features: Union[torch.Tensor, np.ndarray]) -> None:
        """记录输入特征的类型和设备信息，用于采样后恢复。

        Args:
            features: 输入特征
        """
        self.features_is_numpy = isinstance(features, np.ndarray)
        if not self.features_is_numpy:
            self.features_device = features.device

    def _restore_type(self, features: torch.Tensor) -> Union[torch.Tensor, np.ndarray]:
        """将采样结果恢复为输入时的类型和设备。

        Args:
            features: 采样后的 torch Tensor

        Returns:
            恢复为原始类型和设备的特征
        """
        if self.features_is_numpy:
            return features.cpu().numpy()
        return features.to(self.features_device)


class GreedyCoresetSampler(BaseSampler):
    """精确贪婪核心集采样器。

    通过计算完整的 N x N 距离矩阵，迭代选取距离当前核心集最远的点。
    该方法能够精确找到代表性子集，但内存开销较大（O(N^2)）。
    """

    def __init__(
        self,
        percentage: float,
        device: torch.device,
        dimension_to_project_features_to=128,
    ):
        """初始化精确贪婪核心集采样器。

        Args:
            percentage: 采样比例
            device: 运行设备
            dimension_to_project_features_to: 特征降维目标维度（Johnson-Lindenstrauss 变换）
        """
        super().__init__(percentage)
        self.device = device
        self.dimension_to_project_features_to = dimension_to_project_features_to

    def _reduce_features(self, features):
        """使用随机线性映射将特征降维。

        如果原始维度已经等于目标维度，则跳过降维。

        Args:
            features: 输入特征 [N, D]

        Returns:
            降维后的特征 [N, target_dim]
        """
        if features.shape[1] == self.dimension_to_project_features_to:
            return features
        # 使用随机线性映射实现 Johnson-Lindenstrauss 降维
        mapper = torch.nn.Linear(
            features.shape[1], self.dimension_to_project_features_to, bias=False
        )
        _ = mapper.to(self.device)
        features = features.to(self.device)
        return mapper(features)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """执行精确贪婪核心集采样。

        Args:
            features: 输入特征 [N, D]

        Returns:
            采样后的特征 [K, D]，K = int(N * percentage)
        """
        if self.percentage == 1:
            return features
        self._store_type(features)
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features)
        reduced_features = self._reduce_features(features)
        sample_indices = self._compute_greedy_coreset_indices(reduced_features)
        features = features[sample_indices]
        return self._restore_type(features)

    @staticmethod
    def _compute_batchwise_differences(
        matrix_a: torch.Tensor, matrix_b: torch.Tensor
    ) -> torch.Tensor:
        """计算两个矩阵间的成对欧氏距离。

        使用向量化计算: ||a - b||^2 = a^2 + b^2 - 2ab

        Args:
            matrix_a: 矩阵 A [N, D]
            matrix_b: 矩阵 B [M, D]

        Returns:
            距离矩阵 [N, M]
        """
        a_times_a = matrix_a.unsqueeze(1).bmm(matrix_a.unsqueeze(2)).reshape(-1, 1)
        b_times_b = matrix_b.unsqueeze(1).bmm(matrix_b.unsqueeze(2)).reshape(1, -1)
        a_times_b = matrix_a.mm(matrix_b.T)

        return (-2 * a_times_b + a_times_a + b_times_b).clamp(0, None).sqrt()

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """执行精确贪婪核心集选择算法。

        算法步骤:
            1. 计算 N x N 距离矩阵
            2. 初始选择距离原点最远的点
            3. 每次迭代选择距离当前核心集最远的点加入
            4. 重复直到达到目标采样数量

        Args:
            features: 降维后的特征 [N, D]

        Returns:
            核心集索引数组 [K]
        """
        # 计算完整的距离矩阵 [N, N]
        distance_matrix = self._compute_batchwise_differences(features, features)
        # 初始距离：每个点到原点的距离
        coreset_anchor_distances = torch.norm(distance_matrix, dim=1)

        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        for _ in range(num_coreset_samples):
            # 选择距离当前核心集最远的点
            select_idx = torch.argmax(coreset_anchor_distances).item()
            coreset_indices.append(select_idx)

            # 更新每个点到核心集的最小距离
            coreset_select_distance = distance_matrix[
                :, select_idx : select_idx + 1  # noqa E203
            ]
            coreset_anchor_distances = torch.cat(
                [coreset_anchor_distances.unsqueeze(-1), coreset_select_distance], dim=1
            )
            coreset_anchor_distances = torch.min(coreset_anchor_distances, dim=1).values

        return np.array(coreset_indices)


class ApproximateGreedyCoresetSampler(GreedyCoresetSampler):
    """近似贪婪核心集采样器。

    通过随机选择起始点并逐点计算距离，避免了完整 N x N 距离矩阵的计算，
    大幅降低内存开销（从 O(N^2) 降至 O(N * K)），但采样时间可能略有增加。

    这是论文中推荐使用的默认采样方法。
    """

    def __init__(
        self,
        percentage: float,
        device: torch.device,
        number_of_starting_points: int = 10,
        dimension_to_project_features_to: int = 128,
    ):
        """初始化近似贪婪核心集采样器。

        Args:
            percentage: 采样比例
            device: 运行设备
            number_of_starting_points: 随机起始点数量（默认 10）
            dimension_to_project_features_to: 特征降维目标维度
        """
        self.number_of_starting_points = number_of_starting_points
        super().__init__(percentage, device, dimension_to_project_features_to)

    def _compute_greedy_coreset_indices(self, features: torch.Tensor) -> np.ndarray:
        """执行近似贪婪核心集选择算法。

        与精确版本的区别：
            - 不计算完整 N x N 距离矩阵
            - 使用随机起始点构建初始近似距离
            - 每次迭代只计算与当前候选点的距离

        Args:
            features: 降维后的特征 [N, D]

        Returns:
            核心集索引数组 [K]
        """
        # 随机选择起始点
        number_of_starting_points = np.clip(
            self.number_of_starting_points, None, len(features)
        )
        start_points = np.random.choice(
            len(features), number_of_starting_points, replace=False
        ).tolist()

        # 计算每个点到起始点的距离作为初始近似
        approximate_distance_matrix = self._compute_batchwise_differences(
            features, features[start_points]
        )
        approximate_coreset_anchor_distances = torch.mean(
            approximate_distance_matrix, axis=-1
        ).reshape(-1, 1)
        coreset_indices = []
        num_coreset_samples = int(len(features) * self.percentage)

        with torch.no_grad():
            for _ in tqdm.tqdm(range(num_coreset_samples), desc="Subsampling..."):
                # 选择距离当前核心集最远的点
                select_idx = torch.argmax(approximate_coreset_anchor_distances).item()
                coreset_indices.append(select_idx)
                # 计算新选点到所有点的距离
                coreset_select_distance = self._compute_batchwise_differences(
                    features, features[select_idx : select_idx + 1]  # noqa: E203
                )
                # 更新每个点到核心集的最小距离
                approximate_coreset_anchor_distances = torch.cat(
                    [approximate_coreset_anchor_distances, coreset_select_distance],
                    dim=-1,
                )
                approximate_coreset_anchor_distances = torch.min(
                    approximate_coreset_anchor_distances, dim=1
                ).values.reshape(-1, 1)

        return np.array(coreset_indices)


class RandomSampler(BaseSampler):
    """随机采样器。

    从特征集合中随机均匀采样指定比例的子集。
    作为基线方法，用于对比核心集采样的效果。
    """

    def __init__(self, percentage: float):
        """初始化随机采样器。

        Args:
            percentage: 采样比例
        """
        super().__init__(percentage)

    def run(
        self, features: Union[torch.Tensor, np.ndarray]
    ) -> Union[torch.Tensor, np.ndarray]:
        """执行随机采样。

        Args:
            features: 输入特征 [N, D]

        Returns:
            随机采样后的特征 [K, D]
        """
        num_random_samples = int(len(features) * self.percentage)
        subset_indices = np.random.choice(
            len(features), num_random_samples, replace=False
        )
        subset_indices = np.array(subset_indices)
        return features[subset_indices]