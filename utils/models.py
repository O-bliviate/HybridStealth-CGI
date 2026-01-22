import torch
import torch.nn as nn
import torchvision


class ParsableLinear(nn.Module):
    """
    可解析恶意模块 (PMM) - 线性层版本

    功能:
    1. 作为一个全连接层，负责将特征映射到 logits。
    2. 包含一个预设的目标矩阵 T (攻击者已知)。
    3. 提供 compute_malicious_loss 方法，强制输出 Z 逼近 T。

    理论基础: HybridStealth-CGI Section 3.3.1 [cite: 35, 44]
    """

    def __init__(self, input_dim, output_dim, target_scale=10.0, device='cpu'):
        super(ParsableLinear, self).__init__()
        # 标准线性层
        self.linear = nn.Linear(input_dim, output_dim)

        # 预设的目标矩阵 T，攻击者已知
        # 放大 scale 可以增加 PMM 损失在总损失中的比重，强化梯度控制
        self.target_matrix = torch.randn(output_dim).to(device) * target_scale
        self.device = device

    def forward(self, x):
        out = self.linear(x)
        return out

    def compute_malicious_loss(self, out):
        """
        计算恶意损失 L_pmm = 1/2 * || Z - T ||_F^2
        """
        # 广播目标矩阵以匹配 Batch Size (B, D_out)
        if out.shape[0] != self.target_matrix.shape[0]:
            target_batch = self.target_matrix.expand_as(out)
        else:
            target_batch = self.target_matrix

        # 计算 Frobenius 范数
        loss = 0.5 * torch.norm(out - target_batch, p='fro') ** 2
        return loss


class MaliciousResNet18(nn.Module):
    """
    集成 PMM 的 ResNet-18 模型

    特点:
    1. 针对 CIFAR-100 (32x32) 修改了第一层卷积和池化层，保留空间特征。
    2. 移除了原始的全连接层，替换为 ParsableLinear (PMM)。
    3. Forward 返回 (output, malicious_loss) 元组。

    理论基础: WWW.md Section 4.2
    """

    def __init__(self, num_classes=100, pmm_enabled=True, device='cpu'):
        super(MaliciousResNet18, self).__init__()
        # 加载标准 ResNet18 骨架 (不预训练)
        self.base = torchvision.models.resnet18(pretrained=False)

        # [关键修改]: 适配 CIFAR-100 的小尺寸输入 (32x32)
        # 原始 ResNet 用于 ImageNet (224x224)，第一层是 7x7 conv, stride 2
        # 这里改为 3x3 conv, stride 1, padding 1，避免信息过早丢失
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 移除 MaxPool，避免对 32x32 图像降采样过快
        self.base.maxpool = nn.Identity()

        # 移除原始 FC 层
        self.base.fc = nn.Identity()

        self.pmm_enabled = pmm_enabled

        # PMM 层作为分类头
        # ResNet18 的 feature dim 是 512
        self.pmm = ParsableLinear(512, num_classes, device=device)

    def forward(self, x):
        # 提取特征
        feat = self.base.conv1(x)
        feat = self.base.bn1(feat)
        feat = self.base.relu(feat)
        feat = self.base.maxpool(feat)  # Identity

        feat = self.base.layer1(feat)
        feat = self.base.layer2(feat)
        feat = self.base.layer3(feat)
        feat = self.base.layer4(feat)

        feat = self.base.avgpool(feat)
        feat = torch.flatten(feat, 1)  # Flatten: [B, 512]

        # 通过 PMM 层
        out = self.pmm(feat)

        malicious_loss = torch.tensor(0.0).to(x.device)
        if self.pmm_enabled:
            malicious_loss = self.pmm.compute_malicious_loss(out)

        return out, malicious_loss


def get_loki_kernel(client_idx, total_clients, in_channels, out_channels):
    """
    LOKI: 生成定制化的卷积核 (Identity Mapping Sets)

    功能:
    为每个客户端分配特定的输出通道切片，使其梯度互不重叠。

    理论基础: LOKI Section 3.4 / WWW.md Section 4.2 [cite: 53, 156]
    """
    # 初始化全零卷积核 [Out, In, K, K]
    kernel = torch.zeros(out_channels, in_channels, 3, 3)

    # 计算该客户端的通道分配 (Split Scaling)
    # 逻辑: 将总输出通道数平均分配给 N 个客户端
    channels_per_client = out_channels // total_clients

    # 防止通道数不够分的情况
    if channels_per_client < 1:
        channels_per_client = 1

    start_ch = (client_idx * channels_per_client) % out_channels
    end_ch = start_ch + channels_per_client

    # 设置中心点为 1.0 (Identity Pass-through)
    # 这确保了输入特征图被"搬运"到特定的输出通道
    for c in range(start_ch, end_ch):
        # 简单的取模映射，确保 input_c 在有效范围内
        input_c = c % in_channels

        # 在卷积核中心位置置 1
        kernel[c, input_c, 1, 1] = 1.0

    return kernel