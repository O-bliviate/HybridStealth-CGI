import torch
import torch.nn as nn
import torchvision


class ParsableLinear(nn.Module):
    """
    [基于统计特性的 PMM]
    不依赖物理隔离，而是通过权重初始化创造 '统计指纹'。
    """

    def __init__(self, input_dim, output_dim, device='cpu'):
        super(ParsableLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

        # 线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

        # 目标矩阵 T (攻击者已知)
        self.target_matrix = torch.randn(output_dim).to(device) * 10.0

    def inject_statistical_fingerprint(self, client_idx, total_clients):
        """
        [关键] 为特定客户端注入统计指纹 (Soft LOKI)

        原理: 将 W_pmm 设计为近似块对角。
        在属于该 Client 的输出列区间内，权重由 target_client 的数据主导。
        """
        # 计算该客户端的"专属频段" (Block)
        block_size = self.output_dim // total_clients
        if block_size < 1: block_size = 1

        start_col = client_idx * block_size
        end_col = start_col + block_size

        with torch.no_grad():
            # 1. 全局背景: 低能量噪声 (模拟其他客户端的干扰)
            self.linear.weight.data.normal_(0, 0.001)

            # 2. 专属频段: 高能量指纹 (Strong Activation)
            # 在这个块内，我们使用高增益的正交初始化，确保梯度能量集中
            # 注意: PyTorch Linear weight shape is [Out, In]
            # 我们要操作的是 output channel (rows in weight matrix)

            fingerprint = torch.empty(end_col - start_col, self.input_dim)
            nn.init.orthogonal_(fingerprint, gain=2.0)  # Gain > 1 创造强信号

            self.linear.weight.data[start_col:end_col, :] = fingerprint.to(self.device)

    def forward(self, x):
        return self.linear(x)

    def compute_malicious_loss(self, out):
        if out.shape[0] != self.target_matrix.shape[0]:
            target_batch = self.target_matrix.expand_as(out)
        else:
            target_batch = self.target_matrix
        return 0.5 * torch.norm(out - target_batch, p='fro') ** 2


class MaliciousModel(nn.Module):
    """
    [回归标准架构]
    不再需要极宽的卷积层，这提高了攻击的隐蔽性。
    依赖 PMM 层的统计特性进行分离。
    """

    def __init__(self, num_clients=100, num_classes=100, pmm_enabled=True, device='cpu'):
        super(MaliciousModel, self).__init__()

        # 使用标准 ResNet18 (或稍作修改适配 CIFAR)
        # 不再动态扩大 Conv1 通道
        self.base = torchvision.models.resnet18(pretrained=False)
        self.base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.base.maxpool = nn.Identity()
        self.base.fc = nn.Identity()

        self.pmm_enabled = pmm_enabled
        self.device = device

        # PMM 层 (ResNet18 feat dim = 512)
        # 为了支持 Soft LOKI 分离，Output Dim 最好 >= Num_Clients * k (k>=1)
        # 如果 num_classes=100, clients=100, 则每人分配 1 个神经元用于分离
        self.pmm = ParsableLinear(512, num_classes, device=device)

    def forward(self, x):
        feat = self.base.conv1(x)
        feat = self.base.bn1(feat)
        feat = self.base.relu(feat)
        feat = self.base.layer1(feat)
        feat = self.base.layer2(feat)
        feat = self.base.layer3(feat)
        feat = self.base.layer4(feat)
        feat = self.base.avgpool(feat)
        feat = torch.flatten(feat, 1)

        logits = self.pmm(feat)

        malicious_loss = torch.tensor(0.0).to(self.device)
        if self.pmm_enabled:
            malicious_loss = self.pmm.compute_malicious_loss(logits)

        # [修正]: 必须返回 malicious_loss，而不是 0.0
        # 同时返回 feat 供 Phase 2 使用
        return logits, malicious_loss, feat


# 辅助函数: 模拟服务器下发指纹模型
def inject_client_fingerprint(model, client_idx, total_clients):
    model.pmm.inject_statistical_fingerprint(client_idx, total_clients)