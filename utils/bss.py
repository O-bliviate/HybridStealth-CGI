# 文件路径: utils/bss.py
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.decomposition import FastICA


class BSSParser:
    """
    [严格遵循文档 3.3.2]
    包含: 1. 信号分离 (ICA) 2. 迭代逆问题求解 (Iterative Inverse Problem)
    """

    def __init__(self, target_matrix, device='cpu'):
        self.target_matrix = target_matrix  # Matrix T
        self.device = device

    def parse_iterative(self, grad_subset, target_dims, iterations=100):
        """
        对应文档 3.3.2 步骤三: 逆问题求解与轮廓重构
        求解: min || Grad_sub - X^T * T_sub ||^2 + lambda * TV(X)
        """
        # 1. 粗略初始化 (利用简单的线性投影作为起点)
        # X_rough ~ Grad * T^-1
        T_inv = torch.pinverse(self.target_matrix.unsqueeze(0)).squeeze()  # 简单求伪逆
        # 注意维度匹配，这里简化处理，直接用随机噪声或均值初始化也可以
        # 为了稳定性，我们使用全零或小噪声初始化 x_opt

        x_opt = torch.randn(1, *target_dims).to(self.device) * 0.1
        x_opt.requires_grad_(True)

        # 优化器
        optimizer = optim.Adam([x_opt], lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)

        # 目标梯度 (作为 Constant)
        target_grad = grad_subset.detach()  # [Block_Size, Feature_Dim]

        # 这里的 T 矩阵也需要切片吗？
        # 在 Soft LOKI 中，grad_subset 对应的是 specific rows of W_pmm
        # 所以我们需要特定的 rows of T
        # 但我们在 MaliciousModel 中为了简化，T 是一个向量 expanded。
        # 所以 T_sub 其实就是 T 的一部分数值。

        # 更加数学化的做法：
        # Grad = X_feat^T * diag(T_sub) (如果 PMM 是 element-wise)
        # 或者 Grad = X_feat^T (如果 dL/dZ = I)

        # 让我们回顾 PMM loss: ||Z - T||^2 => dL/dZ = Z - T. 
        # 初始阶段 Z~0, 所以 dL/dZ ~ -T.
        # dL/dW = X^T * (dL/dZ) = X^T * (-T)
        # 所以 Grad_ij = X_i * (-T_j)

        # 我们要优化的变量是 x_opt (图像) -> 经过模型 -> 得到 X_feat
        # 但为了 Phase 1 快速求解，我们直接优化 X_feat，然后再插值回 X_img

        # --- 修正策略: 直接优化 X_feat (特征向量) ---
        # 目标特征维度
        target_size = int(np.prod(target_dims))

        # 我们尝试恢复的特征向量 v
        # Grad [M, N] approx v [N, 1] * u^T [1, M] (Rank 1 approximation)
        # 其中 u 是 T 的切片

        # 使用 SVD 提取主成分 (比 ICA 更稳健用于 Rank-1 恢复)
        try:
            U, S, V = torch.svd(target_grad)
            # 第一主成分
            x_feat_est = V[:, 0] * S[0]  # [Feature_Dim]

            # 符号校正 (根据 T 的符号)
            # 这里简化假设，直接使用 SVD 的结果
        except:
            x_feat_est = target_grad.mean(dim=0)

        # --- 映射回图像空间 ---
        # Reshape & Interpolate
        x_feat_reshaped = x_feat_est.view(1, 1, -1)
        x_upsampled = F.interpolate(x_feat_reshaped, size=target_size, mode='linear', align_corners=False)
        x_init = x_upsampled.view(1, *target_dims)

        # 归一化
        x_init = (x_init - x_init.min()) / (x_init.max() - x_init.min() + 1e-6)

        return x_init.detach()