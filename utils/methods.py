import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from utils.data_processing import set_idx, label_mapping
import numpy as np
from .game import NashMSFL
from utils.data_processing import total_variation
from torchmetrics.image import TotalVariation
from utils.save import save_img, early_stop, save_final_img, save_eval
from torch.nn.utils import clip_grad_norm_
from .method_utils import gradient_closure, gradient_closure2
from .method_utils import defense_alg


def mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path,
         str_time):
    """
    标准的 mDLG (同任务) 反演流程。
    适用于 CGI-S 场景，或者简单的多服务器梯度平均场景。
    """
    # 定义损失函数，这里用于计算模型预测的 CrossEntropy，从而求梯度
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1

    # ----------------------------------------------------------------
    # 1. 准备 Ground Truth (真实数据)
    # ----------------------------------------------------------------
    for imidx in range(num_dummy):
        # 获取随机的图片索引
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)

        # 获取真实图片数据，并转为 Tensor
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())  # 增加 batch 维度: [1, C, H, W]

        # 获取真实标签
        tmp_label = torch.Tensor([dst[idx][1]]).long().to(device)
        tmp_label = tmp_label.view(1, )

        # 如果 batch_size > 1，将数据拼接起来
        if imidx == 0:
            gt_data = tmp_datum
            gt_label = tmp_label
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            gt_label = torch.cat((gt_label, tmp_label), dim=0)

    # ----------------------------------------------------------------
    # 2. 准备数据归一化参数
    # ----------------------------------------------------------------
    d_mean, d_std = mean_std
    # 将均值和方差转为 Tensor，用于后续将 dummy_data 限制在有效像素范围内
    dm = torch.as_tensor(d_mean, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()
    ds = torch.as_tensor(d_std, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()

    # ----------------------------------------------------------------
    # 3. 计算真实梯度 (Ground Truth Gradients)
    # ----------------------------------------------------------------
    original_dy_dxs = []
    _label_preds = []

    for i in range(args.num_servers):
        # 如果没有防御策略 (defense_method == 'none')
        if args.defense_method == 'none':
            out = nets[i](gt_data)  # 前向传播：真实图片 -> 模型 -> 输出
            y = criterion(out, gt_label)  # 计算损失
            dy_dx = torch.autograd.grad(y, nets[i].parameters())  # 反向传播：计算对模型参数的梯度
            # 复制梯度，避免后续操作影响原始梯度
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        else:
            # 如果有防御策略（如加噪、剪枝），使用 defense_alg 处理
            original_dy_dx = defense_alg(nets[i], gt_data, gt_label, criterion, device, args)

        original_dy_dxs.append(original_dy_dx)

        # ------------------------------------------------------------
        # 3.1 标签恢复 (Label Recovery / iDLG)
        # ------------------------------------------------------------
        # 利用全连接层梯度的性质恢复标签。
        # 梯度的符号通常由 (prediction - label) 决定，正确类别的梯度通常为负且绝对值最大。
        # original_dy_dx[-2] 通常是最后一层权重的梯度。
        _label_preds.append(
            torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False))

    # 整理恢复出的标签
    label_preds = []
    for i in _label_preds:
        j = i.repeat(args.num_dummy)
        label_preds.append(j)

    # ----------------------------------------------------------------
    # 4. 初始化 Dummy Data (虚拟数据/假图)
    # ----------------------------------------------------------------
    # 使用标准正态分布随机初始化，形状与真实数据一致
    # requires_grad_(True) 非常关键，因为我们要对它求导并更新它
    dummy_data = torch.randn(gt_data.size(), dtype=next(nets[0].parameters()).dtype).to(device).requires_grad_(True)

    # 将初始化的噪声数据投影到有效的图像空间（例如反归一化后在 [0,1] 之间）
    with torch.no_grad():
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

    # 初始化 Dummy Label (如果需要优化标签的话，但在 iDLG 中通常直接使用恢复的标签)
    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

    # ----------------------------------------------------------------
    # 5. 配置优化器
    # ----------------------------------------------------------------
    if args.num_dummy > 1:
        # 如果 Batch Size > 1，通常同时优化数据和标签（DLG 原始做法）
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, dummy_label], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.iteration // 12.0],
                                                             gamma=0.0001)
        else:
            optimizer = optimizer = torch.optim.Adam([dummy_data, dummy_label], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[args.iteration // 1.5], gamma=0.1)
    else:
        # 如果 Batch Size = 1，通常只优化数据，标签使用恢复出的 label_preds
        if args.optim == 'LBFGS':
            optimizer = torch.optim.LBFGS([dummy_data, ], lr=args.lr)
            # 这里的 milestones 设置得比较早 (iteration // 12)，可能导致过早衰减
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 12.0],
                                                             gamma=0.0001)
        else:
            optimizer = torch.optim.Adam([dummy_data, ], lr=args.lr)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                             milestones=[args.iteration // 1.5], gamma=0.1)

    history = []
    history_iters = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)

    # ----------------------------------------------------------------
    # 6. 迭代优化循环 (Reconstruction Loop)
    # ----------------------------------------------------------------
    for iters in range(Iteration):

        # 定义 closure 函数，这是 LBFGS 优化器要求的，用于多次计算损失
        def closure():
            optimizer.zero_grad()  # 清空上一步的梯度

            dummy_dy_dxs = []
            for i in range(args.num_servers):
                # 前向传播：Dummy Data -> 模型 -> 预测
                pred = nets[i](dummy_data)

                # 计算 Dummy Loss
                if args.num_dummy > 1:
                    # DLG 原始做法：同时优化 label，使用 KL 散度或类似
                    dummy_loss = - torch.mean(
                        torch.sum(torch.softmax(dummy_label, -1) * torch.log(torch.softmax(pred, -1)), dim=1))
                else:
                    # iDLG 做法：使用恢复出的 label_preds 计算 CrossEntropy
                    dummy_loss = criterion(pred, label_preds[i])

                # 计算 Dummy Gradient (对模型参数的梯度)
                # create_graph=True 是必须的，因为我们要对“梯度”再求导（二阶导数）
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

            # 计算梯度差异 (Gradient Distance)
            grad_diff = 0
            for i in range(args.num_servers):
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    if args.inv_loss == 'l1':
                        grad_diff += (torch.abs(gx - gy)).sum()
                    else:
                        # 默认 L2 距离：sum((g_dummy - g_real)^2)
                        # 注意：这里 mDLG 简单地将所有服务器的 loss 相加 (等权聚合)
                        grad_diff += ((gx - gy) ** 2).sum()


            # 反向传播：计算 grad_diff 对 dummy_data 的梯度
            grad_diff.backward()
            return grad_diff

        # 执行一步优化更新
        if args.inv_loss == 'l2' or args.inv_loss == 'l1':
            current_loss = optimizer.step(closure)
        else:  # 'sim' (余弦相似度)
            # 如果使用余弦相似度，调用 gradient_closure2
            current_loss = optimizer.step(gradient_closure2(optimizer, dummy_data, original_dy_dxs,
                                                            label_preds, nets, args, criterion))

        # 学习率衰减
        if args.scheduler:
            scheduler.step()

        # 投影回图像空间 (Box Constraint)
        with torch.no_grad():
            dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if (iters + 1 == args.iteration) or iters % 500 == 0:
                print(f'It: {iters}. Rec. loss: {current_loss.item():2.4f}.')

        train_iters.append(iters)

        # ------------------------------------------------------------
        # 7. 记录与评估
        # ------------------------------------------------------------
        if iters % args.log_metrics_interval == 0 or iters in [10 * i for i in range(10)]:
            # 计算 MSE, PSNR, SSIM, LPIPS
            result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0],
                             result[1], result[2], result[3])
            res = [iters]
            res.extend(result)
            results.append(res)

        # 早停机制
        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        # 保存中间结果图片
        if iters % int(Iteration / args.log_interval) == 0 or (args.log_interval == 1 and iters == Iteration - 1):
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")
    return imidx_list, final_iter, final_img, results


def mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst, mean_std, nets, num_classes, Iteration, save_path,
            str_time):
    """
    多任务 mDLG (CGI-D) 反演流程。
    支持 Nash 博弈聚合 (Game Aggregation)。
    """
    criterion = nn.CrossEntropyLoss().to(device)
    imidx_list = []
    final_img = []
    final_iter = Iteration - 1
    tmp_labels = []
    gt_labels = []

    d_mean, d_std = mean_std
    dm = torch.as_tensor(d_mean, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()
    ds = torch.as_tensor(d_std, dtype=next(nets[0].parameters()).dtype)[:, None, None].cuda()

    # ----------------------------------------------------------------
    # 1. 准备 Ground Truth (支持多任务标签)
    # ----------------------------------------------------------------
    for imidx in range(num_dummy):
        tmp_labels = []
        idx, imidx_list = set_idx(imidx, imidx_list, idx_shuffle)
        tmp_datum = tt(dst[idx][0]).float().to(device)
        tmp_datum = tmp_datum.view(1, *tmp_datum.size())

        # 第一个任务的标签
        tmp_labels.append(torch.Tensor([dst[idx][1]]).long().to(device))

        '''获取同一张图片在不同任务下的标签 (Label Mapping)'''
        if args.num_servers > 1:
            for i in range(args.num_servers - 1):
                # 例如：任务1是识别物体，任务2是识别颜色，标签不同
                tmp_labels.append(label_mapping(origin_label=dst[idx][1], idx=i).to(device))

        for i in range(args.num_servers):
            tmp_labels[i] = tmp_labels[i].view(1, )

        if imidx == 0:
            gt_data = tmp_datum
            for i in range(args.num_servers):
                gt_labels.append(tmp_labels[i])
        else:
            gt_data = torch.cat((gt_data, tmp_datum), dim=0)
            for i in range(args.num_servers):
                gt_labels[i] = torch.cat((gt_labels[i], tmp_labels[i]), dim=0)

    # ----------------------------------------------------------------
    # 2. 计算真实梯度
    # ----------------------------------------------------------------
    original_dy_dxs = []
    label_preds = []
    for i in range(args.num_servers):
        if args.defense_method == 'none':
            out = nets[i](gt_data)
            y = criterion(out, gt_labels[i])  # 注意：这里用了 gt_labels[i]，每个任务标签可能不同
            dy_dx = torch.autograd.grad(y, nets[i].parameters())
            original_dy_dx = list((_.detach().clone() for _ in dy_dx))
        else:
            original_dy_dx = defense_alg(nets[i], gt_data, gt_labels[i], criterion, device, args)

        original_dy_dxs.append(original_dy_dx)

        # 标签恢复
        label_preds.append(
            torch.argmin(torch.sum(original_dy_dx[-2], dim=-1), dim=-1).detach().reshape((1,)).requires_grad_(False))
        label_preds[i] = label_preds[i].repeat(args.num_dummy)

    # ----------------------------------------------------------------
    # 3. 初始化 Dummy Data
    # ----------------------------------------------------------------
    dummy_data = torch.randn(gt_data.size(), dtype=next(nets[0].parameters()).dtype).to(device).requires_grad_(True)

    with torch.no_grad():
        dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

    dummy_label = torch.randn((gt_data.shape[0], num_classes)).to(device).requires_grad_(True)

    # ----------------------------------------------------------------
    # 4. 配置优化器
    # ----------------------------------------------------------------
    if args.optim == 'LBFGS':
        optimizer = torch.optim.LBFGS([dummy_data, ], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([dummy_data, ], lr=args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[args.iteration // 2.0], gamma=0.1)

    history = []
    history_iters = []
    train_iters = []
    results = []
    args.logger.info('lr = #{}', args.lr)

    # ----------------------------------------------------------------
    # 5. 迭代优化循环 (含 Nash 聚合)
    # ----------------------------------------------------------------
    for iters in range(Iteration):

        def closure():
            optimizer.zero_grad()
            single_alpha = torch.FloatTensor([0, 1])
            _ = torch.rand(1)
            random_alpha = torch.FloatTensor([_, 1 - _])

            # 计算每个服务器的 Dummy Gradient
            dummy_dy_dxs = []
            for i in range(args.num_servers):
                pred = (nets[i](dummy_data))
                dummy_loss = criterion(pred, label_preds[i])
                dummy_dy_dxs.append(torch.autograd.grad(dummy_loss, nets[i].parameters(), create_graph=True))

            # 计算每个服务器的 Loss (L2 距离)
            losses = []
            for i in range(args.num_servers):
                _loss = 0
                for gx, gy in zip(dummy_dy_dxs[i], original_dy_dxs[i]):
                    _loss += ((gx - gy) ** 2).sum()
                losses.append(_loss)

            # --------------------------------------------------------
            # 5.1 梯度聚合策略 (Aggregation Strategy)
            # --------------------------------------------------------
            if args.diff_task_agg == 'game':
                # 【核心】Nash 博弈聚合
                # 实例化 NashMSFL 求解器
                game = NashMSFL(n_tasks=args.num_servers)
                # 计算最优权重 game_alpha
                _, _, game_alpha = game.get_weighted_loss(losses=losses, dummy_data=dummy_data)
                # 归一化权重
                game_alpha = [game_alpha[i] / sum(game_alpha) for i in range(len(game_alpha))]
                # 加权求和得到最终的梯度差异
                grad_diff = sum([losses[i] * game_alpha[i] for i in range(len(game_alpha))])

            elif args.diff_task_agg == 'single':
                # 只优化其中一个任务
                grad_diff = sum([losses[i] * single_alpha[i] for i in range(len(single_alpha))])

            elif args.diff_task_agg == 'random':
                # 随机权重聚合
                grad_diff = sum([losses[i] * random_alpha[i] for i in range(len(random_alpha))])

            grad_diff.backward()
            return grad_diff

        if args.inv_loss == 'l2':
            current_loss = optimizer.step(closure)
        else:  # 'sim'
            current_loss = optimizer.step(gradient_closure2(optimizer, dummy_data, original_dy_dxs,
                                                            label_preds, nets, args, criterion))
        if args.scheduler:
            scheduler.step()

        with torch.no_grad():
            dummy_data.data = torch.max(torch.min(dummy_data, (1 - dm) / ds), -dm / ds)

            if (iters + 1 == args.iteration) or iters % 500 == 0:
                print(f'It: {iters}. Rec. loss: {current_loss.item():2.4f}.')

        train_iters.append(iters)

        if iters % args.log_metrics_interval == 0 or iters in [10 * i for i in range(10)]:
            result = save_eval(args.get_eval_metrics(), dummy_data, gt_data)
            args.logger.info('iters idx: #{}, current lr: #{}', iters, optimizer.param_groups[0]['lr'])
            args.logger.info('loss: #{}, mse: #{}, lpips: #{}, psnr: #{}, ssim: #{}', current_loss, result[0],
                             result[1], result[2], result[3])
            res = [iters]
            res.extend(result)
            results.append(res)

        if args.earlystop > 0:
            _es = early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy,
                             save_path)
            if _es:
                break

        if iters % int(Iteration / args.log_interval) == 0:
            save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list,
                     str_time, mean_std)

    args.logger.info("inversion finished")

    return imidx_list, final_iter, final_img, results