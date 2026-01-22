import torch
import torch.nn.functional as F
import numpy as np


class BSSParser:
    """
    盲源分离与解析器 (Blind Source Separation & Parsing)
    对应 HybridStealth Phase 1: 解析初始化
    """

    def __init__(self, target_matrix, device='cpu'):
        """
        Args:
            target_matrix: PMM层中预设的目标矩阵 T (D_out,)
        """
        self.target_matrix = target_matrix
        self.device = device

    def parse_x_low(self, grad_pmm_weight, target_dims):
        """
        解析出数据的低频轮廓 X_low
        Args:
            grad_pmm_weight: PMM层的权重梯度 (D_out, D_in)
            target_dims: 目标重构数据的维度 (C, H, W) -> (3, 32, 32)
        Returns:
            x_low: 解析出的低频轮廓 (1, C, H, W)
        """
        # 1. 理论解析: X_feat^T = Grad @ C_inv
        # 简化假设: C 近似为 Target Matrix T
        T_reciprocal = 1.0 / (self.target_matrix + 1e-6)

        # [D_in, D_out] @ [D_out, D_out] -> [D_in, D_out]
        # 这里的 D_in 是 512 (Feature Dim)
        x_low_est = torch.matmul(grad_pmm_weight.t(), torch.diag(T_reciprocal))

        # 2. 聚合得到特征向量
        # [D_in, D_out] -> [D_in] (即 512 维特征)
        x_feat = x_low_est.sum(dim=1)  # Shape: [512]

        # 3. [关键修复]: 维度适配与重塑
        # 目标总维度
        target_size = int(np.prod(target_dims))  # 3072
        current_size = x_feat.numel()  # 512

        if current_size == target_size:
            # 如果维度正好匹配 (比如 PMM 在第一层)，直接 Reshape
            x_low = x_feat.view(1, *target_dims)
        else:
            # 如果维度不匹配 (PMM 在特征层)，说明我们解析的是特征
            # 我们需要将其"上采样"或"映射"回图像空间作为初始化

            # [关键修改]: Reshape 成 [1, 1, 512] (3D Tensor)
            # PyTorch 的 linear 插值只支持 3D 输入 (N, C, L)
            # 这样 PyTorch 就知道我们是在对 1D 序列进行拉伸
            x_feat_reshaped = x_feat.view(1, 1, -1)

            # 强制插值到目标总像素数 [1, 1, 3072]
            x_upsampled = F.interpolate(x_feat_reshaped, size=target_size, mode='linear', align_corners=False)

            # Reshape 成目标图像形状 [1, 3, 32, 32]
            x_low = x_upsampled.view(1, *target_dims)

            # 可选: 对结果进行归一化，防止数值过大
            x_low = (x_low - x_low.min()) / (x_low.max() - x_low.min() + 1e-6)
            # 映射回 [-1, 1] 区间
            x_low = x_low * 2 - 1

        # 截断异常值
        x_low = torch.clamp(x_low, min=-2, max=2)

        return x_low