import matplotlib.pyplot as plt
from utils.evaluation import PSNR, SSIM, LPIPS
import torch
import os
import csv


def save_img(iters, args, history, tp, dummy_data, num_dummy, history_iters, gt_data, save_path, imidx_list, str_time, mean_std=None):
    if not mean_std is None:
        d_mean, d_std = mean_std
        dm = torch.as_tensor(d_mean)[:, None, None].cuda()
        ds = torch.as_tensor(d_std)[:, None, None].cuda()
        dummy_data = torch.clamp(dummy_data * ds + dm, 0, 1)
        gt_data = torch.clamp(gt_data * ds + dm, 0, 1)

    history.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
    history_iters.append(iters)

    for imidx in range(num_dummy):
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 7, 1)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.imshow(tp(gt_data[imidx].cpu()))
        plt.title('original')
        plt.axis('off')
        for i in range(min(len(history), 6)):
            plt.subplot(1, 7, i + 2)
            plt.imshow(history[i][imidx])
            plt.title('iter=%d' % (history_iters[i]))
            plt.axis('off')

        method_name = args.methods[0]
        if method_name == 'mDLG_mt':
            method_name = method_name + '_' + args.diff_task_agg

        plt.savefig('%s/%s_on_%s_%s_%s_%s_%s_%s.png' % (
        save_path, method_name, imidx_list[imidx], args.get_dataset(), args.get_net(), args.num_servers, str_time, args.defense_method))
        plt.close()

def save_final_img(iters, final_iter,final_img, tp, dummy_data, imidx, num_dummy, save_path, args, imidx_list):
    if iters == final_iter:
        print('this is final iter')
        final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        plt.imshow(final_img[0][imidx])
        plt.title('iter=%d' % (final_iter))
        plt.axis('off')
        plt.savefig('%s/%s_final_on_%s_%05d.png' % (save_path, args.methods[0], imidx_list, imidx_list[imidx]))
        plt.close()
        args.logger.info('final iter: #{}', final_iter)

def early_stop(args, current_loss, iters, tp, final_img, dummy_data, imidx_list, imidx, num_dummy, save_path):
    # converge
    if abs(current_loss) < args.get_earlystop():
        final_iter = iters
        print('this is final iter')
        final_img.append([tp(dummy_data[imidx].cpu()) for imidx in range(num_dummy)])
        plt.imshow(final_img[0][imidx])
        plt.title('iter=%d' % (final_iter))
        plt.axis('off')
        plt.savefig('%s/%s_final_on_%s.png' % (save_path, args.methods[0], imidx_list[imidx]))
        plt.close()
        return True
    else:
        return False

def save_eval(eval_metrics, dummy_data, gt_data):
    mse = psnr = ssim = lpips = 0
    # Detach tensors before evaluation to prevent warnings and potential side effects
    dummy_data_eval = dummy_data.detach()
    gt_data_eval = gt_data.detach()

    if 'mse' in eval_metrics:
        mse = (torch.mean((dummy_data_eval - gt_data_eval) ** 2).item())
    if 'psnr' in eval_metrics:
        psnr = float(PSNR(dummy_data_eval, gt_data_eval))
    if 'ssim' in eval_metrics:
        ssim = float(SSIM(dummy_data_eval, gt_data_eval))
    if 'lpips' in eval_metrics:
        val = LPIPS(dummy_data_eval, gt_data_eval)
        if torch.is_tensor(val):
            lpips = val.mean().item()
        else:
            lpips = float(val)

    return [mse, lpips, psnr, ssim]

def save_results(results, filename, args=None):
    """
    :param results: experiment results
    :type results: list()
    :param filename: File name to write results to
    :type filename: String
    """
    if args is None:
        dirname = 'debug_results/eval_res'
    else:
        dirname = args.debugOrRun + '/eval_res'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(os.path.join(dirname,filename), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        for experiment in results:
            writer.writerow(experiment)