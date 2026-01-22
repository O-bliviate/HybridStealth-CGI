import torch
import torch.nn as nn

class ParsableLinear(nn.Module):
    """
    可解析恶意模块 (PMM) - 线性层版本
    理论基础: HybridStealth-CGI Section 3.3.1
    """
    def __init__(self, input_dim, output_dim, target_scale=10.0, device='cpu'):
        super(ParsableLinear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        # 预设的目标矩阵 T，攻击者已知
        self.target_matrix = torch.randn(output_dim).to(device) * target_scale
        self.device = device

    def forward(self, x):
        out = self.linear(x)
        return out

    def compute_malicious_loss(self, out):
        """
        计算恶意损失 L_pmm = 1/2 * || Z - T ||_F^2
        """
        # 广播目标矩阵以匹配Batch Size (B, D_out)
        target_batch = self.target_matrix.expand_as(out)
        loss = 0.5 * torch.norm(out - target_batch, p='fro')**2
        return loss

def get_loki_kernel(client_idx, total_clients, in_channels, out_channels):
    """
    LOKI: 生成定制化的卷积核 (Identity Mapping Sets)
    理论基础: LOKI Section 3.4
    功能: 为每个客户端分配特定的输出通道切片，使其梯度互不重叠。
    """
    # 初始化全零卷积核
    kernel = torch.zeros(out_channels, in_channels, 3, 3)

    # 计算该客户端的通道分配 (Split Scaling)
    channels_per_client = out_channels // total_clients
    if channels_per_client < 1:
        channels_per_client = 1  # 简化处理，防止除零

    start_ch = (client_idx * channels_per_client) % out_channels
    end_ch = start_ch + channels_per_client

    # 设置中心点为1 (Identity Pass-through)
    # 使得 Input_c -> Output_c 的映射仅在特定区间激活
    for c in range(start_ch, end_ch):
        input_c = c % in_channels
        kernel[c, input_c, 1, 1] = 1.0

    return kernel