from loguru import logger
import arguments
import os
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

# [关键修改]: 适配 Soft LOKI (统计指纹)
# 移除了 get_loki_kernel，引入 inject_client_fingerprint
from utils.models import MaliciousModel, inject_client_fingerprint
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

    if not os.path.exists(save_path):
        os.makedirs(save_path)

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
    # 强制设置 N=100 以匹配实验设计
    args.num_clients = 100

    tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle, mean_std = load_data(
        dataset=dataset_name, root_path=root_path, data_path=data_path, save_path=save_path)

    ''' train DLG and iDLG and mDLG and DLGAdam'''
    for idx_net in range(num_exp):

        args.logger.info('running #{}|#{} experiment', idx_net, num_exp)

        '''train on different methods'''
        for method in methods:
            args.logger.info('#{}, Try to generate #{} images', method, num_dummy)

            # =================================================================
            # [修改部分]: HybridStealth 100客户端批量仿真 (Soft LOKI版)
            # =================================================================
            if method == 'HybridStealth':
                args.logger.info("Initializing HybridStealth-CGI Batch Simulation...")

                # 1. 初始化恶意全局模型 (标准 ResNet + PMM)
                global_model = MaliciousModel(num_clients=args.num_clients,
                                              num_classes=num_classes,
                                              pmm_enabled=True,
                                              device=device).to(device)

                # 容器: 用于存储所有客户端梯度的总和 (模拟安全聚合)
                aggregated_gradients_sum = {}

                # 容器: 存储所有客户端的真实数据 (Ground Truth) 用于后续评估
                gt_data_list = []

                # 起始索引
                start_img_idx = args.get_imidx()

                args.logger.info(f">>> Simulating Secure Aggregation for {args.num_clients} Clients...")

                # 2. 循环模拟 100 个客户端
                for i in range(args.num_clients):
                    # A. 准备数据
                    current_data_idx = idx_shuffle[i]
                    gt_img = tt(dst[current_data_idx][0]).float().to(device).unsqueeze(0)
                    gt_label = torch.tensor([dst[current_data_idx][1]]).long().to(device)
                    gt_data_list.append(gt_img)

                    # B. [关键修改] Soft LOKI: 注入统计指纹
                    # 不再修改卷积核，而是修改 PMM 层的权重分布
                    # 这会让 Client i 的数据在 PMM 输出的特定频段产生强激活
                    inject_client_fingerprint(global_model, i, args.num_clients)

                    # C. 本地训练与梯度计算
                    optimizer = optim.SGD(global_model.parameters(), lr=args.lr)
                    optimizer.zero_grad()

                    output, pmm_loss, _ = global_model(gt_img)
                    cls_loss = F.cross_entropy(output, gt_label)

                    # 组合损失
                    total_loss = cls_loss + args.pmm_scale * pmm_loss
                    total_loss.backward()

                    # D. 安全聚合 (累加)
                    for name, param in global_model.named_parameters():
                        if param.grad is not None:
                            if name not in aggregated_gradients_sum:
                                aggregated_gradients_sum[name] = torch.zeros_like(param.grad)
                            aggregated_gradients_sum[name] += param.grad.detach()

                    if (i + 1) % 10 == 0:
                        args.logger.info(f"   Processed Client {i + 1}/{args.num_clients}")

                args.logger.info("Secure Aggregation Completed. Gradients Captured.")

                # 构造梯度字典
                captured_gradients = {
                    'server_1': aggregated_gradients_sum,
                    'server_2': aggregated_gradients_sum
                }

                # 3. 执行批量攻击
                attacker = HybridStealthAttacker(args, global_model, device)

                # 调用批量重构接口
                rec_imgs, leak_rate, avg_psnr = attacker.reconstruct_all(
                    aggregated_gradients=captured_gradients,
                    gt_data_list=gt_data_list
                )

                # 将 leakage_rate 改为 leak_rate
                args.logger.info(f"Batch Attack Finished. Leakage Rate: {leak_rate:.2f}%, Avg PSNR: {avg_psnr:.2f}")

                # 4. 保存结果
                final_results = [[Iteration, 0.0, 0.0, avg_psnr, 0.0]]  # 仅记录 Avg PSNR

                # 保存 CSV
                save_results(final_results,
                             root_path + '/HybridStealth_Batch_N' + str(args.num_clients) + '.csv',
                             args)

                # 保存可视化
                if len(rec_imgs) > 0:
                    vis_list = []
                    count = min(8, len(rec_imgs))
                    for k in range(count):
                        vis_list.append(gt_data_list[k].detach().cpu())
                        vis_list.append(rec_imgs[k])

                    grid = torchvision.utils.make_grid(torch.cat(vis_list), nrow=2, padding=2, normalize=True)
                    save_image_path = save_path + f"/batch_attack_N{args.num_clients}.png"
                    torchvision.utils.save_image(grid, save_image_path)
                    args.logger.info(f"Result image saved to {save_image_path}")

            # =================================================================
            # 原有方法保留
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

    logger.remove(handler)


if __name__ == '__main__':
    main()