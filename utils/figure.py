import matplotlib.pyplot as plt
import os
import time

def plot_metrics(results, args, save_path):
    """
    Plot the evaluation metrics (MSE, LPIPS, PSNR, SSIM) over iterations.
    
    Args:
        results (list): List of results, where each item is [iteration, mse, lpips, psnr, ssim].
        args (Arguments): The arguments object containing experiment metadata.
        save_path (str): The directory to save the figure.
    """
    if not results:
        print("No results to plot.")
        return

    # Unpack the results
    # Structure assumed: [iteration, mse, lpips, psnr, ssim]
    iters = [row[0] for row in results]
    mse_list = [row[1] for row in results]
    lpips_list = [row[2] for row in results]
    psnr_list = [row[3] for row in results]
    ssim_list = [row[4] for row in results]

    # Create a figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Set the main title with experiment details
    title_str = (f"Experiment Metrics\n"
                 f"Dataset: {args.dataset} | Network: {args.net} | Optimizer: {args.optim} | Methods: {args.methods} \n"
                 f"LR: {args.lr} | Image ID: {args.set_imidx}")
    fig.suptitle(title_str, fontsize=16, fontweight='bold')

    # Plot MSE
    axs[0, 0].plot(iters, mse_list, 'r-', linewidth=2)
    axs[0, 0].set_title('MSE (Lower is Better)', fontsize=14)
    axs[0, 0].set_xlabel('Iteration')
    axs[0, 0].set_ylabel('MSE Value')
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot LPIPS
    axs[0, 1].plot(iters, lpips_list, 'g-', linewidth=2)
    axs[0, 1].set_title('LPIPS (Lower is Better)', fontsize=14)
    axs[0, 1].set_xlabel('Iteration')
    axs[0, 1].set_ylabel('LPIPS Value')
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)

    # Plot PSNR
    axs[1, 0].plot(iters, psnr_list, 'b-', linewidth=2)
    axs[1, 0].set_title('PSNR (Higher is Better)', fontsize=14)
    axs[1, 0].set_xlabel('Iteration')
    axs[1, 0].set_ylabel('PSNR (dB)')
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)

    # Plot SSIM
    axs[1, 1].plot(iters, ssim_list, 'm-', linewidth=2)
    axs[1, 1].set_title('SSIM (Higher is Better)', fontsize=14)
    axs[1, 1].set_xlabel('Iteration')
    axs[1, 1].set_ylabel('SSIM Value')
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)

    # Adjust layout to prevent overlap
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])

    # Construct filename
    timestamp = int(time.time())
    filename = f"metrics_{args.dataset}_{args.net}_{args.optim}_{args.set_imidx}_{timestamp}.png"
    full_path = os.path.join(save_path, filename)

    # Save the figure
    plt.savefig(full_path, dpi=300)
    plt.close()
    
    print(f"Metrics figure saved to: {full_path}")
