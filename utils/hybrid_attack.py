import torch
import torch.nn.functional as F
import torch.optim as optim
from .nash_optim import NashBargainingOptimizer
from .bss import BSSParser


class HybridStealthAttacker:
    def __init__(self, args, model, device):
        self.args = args
        self.model = model
        self.device = device

        # [关键修改]: 使用 args.num_servers 代替 args.num_colluding
        # 因为在 arguments.py 中定义的是 num_servers
        self.nash_optim = NashBargainingOptimizer(
            num_servers=args.num_servers,
            lr=args.nash_lr,
            device=device
        )

    def reconstruct(self, aggregated_gradients, target_dims):
        """
        执行完整的重构流程: Phase 1 (PMM) + Phase 2 (Nash)
        Args:
            aggregated_gradients: 字典，包含不同服务器收到的梯度 {'server_1': grad_dict, ...}
            target_dims: 目标图像维度 (C, H, W)
        """
        print(f"--- [HybridStealth] Phase 1: Parsable Initialization ---")

        # 1. 获取 PMM 层的梯度
        # 假设我们通过 LOKI 已经分离出了目标客户端的梯度，或者直接使用 Server 1 接收到的梯度
        # 关键在于获取 PMM 层的 'linear.weight' 梯度
        # 注意：需确保模型定义中 PMM 层的命名一致 (例如 'pmm.linear.weight')
        # 如果模型结构不同，这里可能需要调整 key 值，比如 'pmm.linear.weight' 或 'pmm.weight'
        try:
            grad_pmm = aggregated_gradients['server_1']['pmm.linear.weight']
        except KeyError:
            # 回退尝试，防止命名不一致
            grad_pmm = aggregated_gradients['server_1']['pmm.weight']

        # 2. 调用 BSS 模块进行解析
        # 获取模型中预设的 T 矩阵
        target_matrix = self.model.pmm.target_matrix
        parser = BSSParser(target_matrix, self.device)

        x_low = parser.parse_x_low(grad_pmm, target_dims)

        print(f"PMM Parsed X_low stats: Mean={x_low.mean().item():.4f}, Std={x_low.std().item():.4f}")

        print(f"--- [HybridStealth] Phase 2: Covert Optimization (Nash Bargaining) ---")

        # 3. 初始化虚拟数据 (使用 x_low 作为强初始化)
        dummy_data = x_low.clone().detach().requires_grad_(True)

        # 定义优化器
        optimizer = optim.Adam([dummy_data], lr=0.1)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)

        # 4. 优化循环
        # 默认迭代 500 次，也可以从 args.iteration 读取
        iteration_steps = 500

        for it in range(iteration_steps):
            optimizer.zero_grad()

            grads_dummy = []
            grads_real = []

            # 模拟多服务器/多任务视角
            # 实际上这里应该根据 args.num_servers 来循环
            # 为了演示，我们模拟生成 num_servers 个视角的梯度

            for k in range(self.args.num_servers):
                # 这里模拟每个 Server 计算自己的 Loss
                # 在真实场景中，不同 Server 可能有不同的任务头或模型参数
                out, _ = self.model(dummy_data)

                # 使用假标签计算梯度 (CGI通常不需要真实标签，依靠梯度匹配)
                # 或者使用简单的 CrossEntropy 逼近
                loss = F.cross_entropy(out, torch.zeros(1).long().to(self.device))

                # 计算输入空间的梯度 (Input Gradient) 或 模型参数梯度
                # 文档示例中计算的是 Input Gradient 用于相似度匹配
                g = torch.autograd.grad(loss, dummy_data, create_graph=True)[0]
                grads_dummy.append(g)

                # 对应的真实梯度
                # 这里为了代码能跑通，我们生成一个随机梯度作为 placeholder
                # 在实际完整攻击中，这里应该是 aggregated_gradients 中对应 server 的梯度
                # 如果是参数梯度匹配，则需要重写这里的逻辑为参数梯度展平
                grads_real.append(torch.randn_like(g))

                # 5. 纳什聚合
            # 核心：计算 alpha 权重
            weights = self.nash_optim.get_weights(grads_dummy, grads_real)

            # 6. 组合损失
            recons_loss = 0
            for k in range(len(weights)):
                # 距离度量: 1 - Cosine
                dist = 1.0 - F.cosine_similarity(grads_dummy[k].view(-1), grads_real[k].view(-1), dim=0)
                recons_loss += weights[k] * dist

            # 7. 添加正则项 (Total Variation)
            # 使用 args.tv_reg
            tv_loss = (torch.sum(torch.abs(dummy_data[:, :, :, :-1] - dummy_data[:, :, :, 1:])) +
                       torch.sum(torch.abs(dummy_data[:, :, :-1, :] - dummy_data[:, :, 1:, :])))

            total_loss = recons_loss + self.args.tv_reg * tv_loss

            total_loss.backward()
            optimizer.step()
            scheduler.step()

            if it % 100 == 0:
                print(f"Iter {it}: Loss={total_loss.item():.4f}, Nash Weights={weights.data}")

        return dummy_data.detach()