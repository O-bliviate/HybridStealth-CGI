from loguru import logger
import arguments
import os
from utils.methods import mdlg, mdlg_mt
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms
from utils import files
from utils.data_download import load_data
from utils.net_utils import intialize_nets
from utils.save import save_results
from utils.figure import plot_metrics

# [关键修改]: 引入我们重构后的模块
from utils.models import MaliciousResNet18, get_loki_kernel  # 假设恶意模型定义在 models.py
from utils.hybrid_attack import HybridStealthAttacker


def main():
    args = arguments.Arguments(logger)

    # Initialize logger
    log_files, str_time = files.files(args)
    handler = logger.add(log_files[0], enqueue=True)

    dataset_name = args.get_dataset()
    root_path = args.get_root_path()
    data_path = os.path.join(root_path, 'data').replace('\\', '/')
    save_path = os.path.join(root_path, args.get_debugOrRun() + '/compare_%s' % dataset_name).replace('\\', '/')

    lr = args.get_lr()
    num_dummy = args.get_num_dummy()
    Iteration = args.get_iteration()
    num_exp = args.get_num_exp()
    methods = args.get_methods()
    log_interval = args.get_log_interval()
    use_cuda = torch.cuda.is_available()
    device = 'cuda' if use_cuda else 'cpu'

    args.log()
    args.logger.info(f'dataset is {dataset_name}')
    args.logger.info(f'device is {device}')

    # 加载数据
    tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle, mean_std = load_data(
        dataset=dataset_name, root_path=root_path, data_path=data_path, save_path=save_path)

    ''' train DLG and iDLG and mDLG and DLGAdam'''
    for idx_net in range(num_exp):

        args.logger.info('running #{}|#{} experiment', idx_net, num_exp)

        '''train on different methods'''
        for method in methods:
            args.logger.info('#{}, Try to generate #{} images', method, num_dummy)

            # =================================================================
            # [修改部分]: HybridStealth 攻击逻辑重构
            # =================================================================
            if method == 'HybridStealth':
                args.logger.info("Initializing HybridStealth-CGI Attack...")

                # 1. 初始化恶意全局模型 (包含 PMM 层)
                # 注意: 我们不使用 intialize_nets，而是使用专门的恶意模型类
                global_model = MaliciousResNet18(num_classes=num_classes, pmm_enabled=True, device=device).to(device)

                # 2. LOKI: 生成并注入恶意卷积核
                # 假设攻击目标是 idx_shuffle[0] 对应的那个逻辑客户端ID
                # 为了简化，我们假设逻辑ID就是 0
                target_client_logic_id = 0
                # 获取第一层卷积的参数维度
                out_channels = global_model.base.conv1.out_channels
                in_channels = global_model.base.conv1.in_channels

                loki_kernel = get_loki_kernel(target_client_logic_id, args.num_clients, in_channels, out_channels)

                # 注入卷积核
                with torch.no_grad():
                    global_model.base.conv1.weight.data = loki_kernel.to(device)
                args.logger.info("LOKI Kernels injected for Target Client #0")

                # 3. 模拟客户端训练与梯度捕获 (Client Side Simulation)
                # 获取真实目标图像和标签
                # idx_shuffle[0] 是数据集中的索引
                gt_img = tt(dst[idx_shuffle[0]][0]).float().to(device).unsqueeze(0)  # [1, C, H, W]
                gt_label = torch.tensor([dst[idx_shuffle[0]][1]]).long().to(device)

                # 前向传播计算 Loss (包含 PMM 恶意损失)
                optimizer = optim.SGD(global_model.parameters(), lr=args.lr)
                optimizer.zero_grad()

                output, pmm_loss = global_model(gt_img)
                cls_loss = F.cross_entropy(output, gt_label)

                # 组合损失: 分类损失 + PMM损失 (pmm_scale 通常较大，如10.0)
                total_loss = cls_loss + args.pmm_scale * pmm_loss
                total_loss.backward()

                # 捕获梯度 (模拟从安全聚合中分离出的梯度)
                # 构造符合 HybridStealthAttacker 接口的字典结构
                # 假设有两个合谋服务器，它们看到的梯度是近似的 (在Input Space匹配时)
                # 关键是捕获 PMM 层的梯度用于 Phase 1 解析
                captured_gradients = {
                    'server_1': {k: v.grad.clone() for k, v in global_model.named_parameters() if v.grad is not None},
                    # 实际场景中 Server 2 的梯度可能略有不同(因任务不同)，这里简化为相同
                    'server_2': {k: v.grad.clone() for k, v in global_model.named_parameters() if v.grad is not None}
                }
                args.logger.info(f"Gradients captured. PMM Loss: {pmm_loss.item():.4f}")

                # 4. 执行攻击 (Phase 1 解析 + Phase 2 优化)
                attacker = HybridStealthAttacker(args, global_model, device)

                # 调用重构
                # [修正]: 关键字必须是 aggregated_gradients (与类定义一致)
                # 实参仍然是 captured_gradients (这是你在 main 中定义的变量名)
                reconstructed_img = attacker.reconstruct(
                    aggregated_gradients=captured_gradients,
                    target_dims=(channel, input_size, input_size)  # (C, H, W)
                )

                # 5. 结果处理与保存
                # 为了兼容 save_results 的接口，我们需要构造一个 results 列表
                # 计算最终指标
                mse = F.mse_loss(reconstructed_img, gt_img).item()
                # 这里简单计算 PSNR
                psnr = 10 * torch.log10(1 / torch.tensor(mse)).item() if mse > 0 else 100

                args.logger.info(f"Attack Finished. Final MSE: {mse:.6f}, PSNR: {psnr:.2f}")

                # 构造 result 列表: [iter, mse, lpips, psnr, ssim]
                # 注意: 这里只有最终结果，没有中间迭代的历史记录，除非 attacker.reconstruct 返回历史
                # 简单起见，我们只存一行
                final_results = [[Iteration, mse, 0.0, psnr, 0.0]]

                # 保存 CSV
                save_results(final_results,
                             root_path + '/HybridStealth_' + str(idx_shuffle[0]) + '_' + args.get_dataset() + '.csv',
                             args)

                # 保存对比图片
                # 将真实图片和重构图片拼接
                comparison = torch.cat([gt_img.detach().cpu(), reconstructed_img.detach().cpu()])
                save_image_path = save_path + f"/hybrid_stealth_{idx_shuffle[0]}.png"
                # 确保目录存在
                if not os.path.exists(save_path): os.makedirs(save_path)
                torchvision.utils.save_image(comparison, save_image_path)
                args.logger.info(f"Result image saved to {save_image_path}")

            # =================================================================
            # 原有方法保留 (mDLG, mDLG_mt 等)
            # =================================================================
            if method == 'mDLG':
                nets = intialize_nets(method=method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                      alter_num_classes=alter_num_classes, input_size=input_size)
                for net in nets:
                    net = net.to(device)
                imidx_list, final_iter, final_img, results = mdlg(args, device, num_dummy, idx_shuffle, tt, tp, dst,
                                                                  mean_std, nets, num_classes, Iteration, save_path,
                                                                  str_time)
                save_results(results, root_path + '/' + method + '_' + str(
                    imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(
                    args.num_servers) + '_' + str_time + '_' + args.defense_method + '.csv', args)
                plot_metrics(results, args, save_path)

            if method == 'mDLG_mt':
                nets = intialize_nets(method=method, args=args, channel=channel, hidden=hidden, num_classes=num_classes,
                                      alter_num_classes=alter_num_classes, input_size=input_size)
                for i in range(len(nets)):
                    nets[i] = nets[i].to(device)
                    args.logger.info('Size of net #{} is #{}', i, len(nets[i].state_dict()))
                imidx_list, final_iter, final_img, results = mdlg_mt(args, device, num_dummy, idx_shuffle, tt, tp, dst,
                                                                     mean_std, nets, num_classes, Iteration, save_path,
                                                                     str_time)
                method = method + '_' + args.diff_task_agg
                save_results(results, root_path + '/' + method + '_' + str(
                    imidx_list[0]) + '_' + args.get_dataset() + '_' + args.get_net() + '_' + str(
                    args.num_servers) + '_' + str_time + '_' + args.defense_method + '.csv', args)
                plot_metrics(results, args, save_path)

        # args.logger.info('imidx_list: #{}', imidx_list)
    logger.remove(handler)


if __name__ == '__main__':
    main()