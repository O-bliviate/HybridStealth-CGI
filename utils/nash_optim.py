import torch
import torch.optim as optim
import torch.nn.functional as F


class NashBargainingOptimizer:
    """
    纳什议价解求解器 (基于 WWW.md / 实验代码生成指导.pdf)
    用于CGI-D场景下不同任务梯度的聚合。
    目标: max sum log(u_k(alpha) - d_k)
    """

    def __init__(self, num_servers, lr=0.01, device='cpu'):
        self.num_servers = num_servers
        self.lr = lr
        self.device = device
        # 聚合权重 alpha，初始化为均等，且需要梯度以进行优化
        self.alpha = torch.ones(num_servers, requires_grad=True, device=device)

    def get_weights(self, dummy_grads, target_grads):
        """
        根据虚拟梯度与目标梯度的相似度，动态计算权重。
        Args:
            dummy_grads: 列表，每个合谋服务器基于当前虚拟数据计算出的梯度
            target_grads: 列表，每个合谋服务器收到的真实梯度
        """
        # 克隆alpha以进行优化更新，避免破坏内部状态的计算图
        alpha_optim = self.alpha.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([alpha_optim], lr=self.lr)

        # 内循环优化求解 Nash Bargaining Problem (文档建议10步)
        for _ in range(10):
            optimizer.zero_grad()

            # 1. 计算效用 Utility u_k
            utilities = []
            for k in range(self.num_servers):
                # [关键修复]: 必须 detach()！
                # 在纳什议价步骤中，我们只优化 alpha (权重)，梯度仅仅是作为观测值(常量)存在。
                # 如果不 detach，loss.backward() 会试图沿图回传到 dummy_data，导致二次反向传播错误。
                g_dummy = dummy_grads[k].view(-1).detach()
                g_target = target_grads[k].view(-1).detach()

                # 使用余弦相似度作为效用度量
                cosine = F.cosine_similarity(g_dummy.unsqueeze(0), g_target.unsqueeze(0)).squeeze()

                # 效用必须大于分歧点 (d_k = 0), 使用 softplus 保证正数及平滑梯度
                u_k = F.softplus(cosine) + 1e-6
                utilities.append(u_k)

            # 2. 归一化权重 (Softmax)
            weights = F.softmax(alpha_optim, dim=0)

            # 3. 计算纳什乘积 (对数和)
            # Objective: Maximize sum(log(u_k_weighted))
            weighted_utility = sum([weights[k] * torch.log(utilities[k]) for k in range(self.num_servers)])

            loss = -weighted_utility  # 梯度下降最小化负效用

            # 这里的 backward 现在只会影响 alpha_optim，因为 utilities 已经被 detach 了
            loss.backward()
            optimizer.step()

        return F.softmax(alpha_optim, dim=0).detach()