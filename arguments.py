import torch.nn.functional as F
import torch
import json
import time

# Setting the seed for Torch
SEED = 1
torch.manual_seed(SEED)


class Arguments:

    def __init__(self, logger):
        self.logger = logger
        self.debugOrRun = 'results'
        self.dataset = 'cifar100'  # 'cifar100', 'lfw', 'mnist', 'celebA', 'stl10'
        self.num_classes = 100
        self.set_imidx = 10080  # int or 000000
        self.net = 'resnet20-4'  # 'lenet', 'resnet20-4', 'resnet34', 'vgg11'
        self.net_mt_diff = True
        self.batch_size = 4
        self.model_path = './model'
        self.root_path = '.'

        self.inv_loss = 'sim'  # Use Cosine Similarity for Phase 2 stability

        self.lr = 0.05  # Very low LR for Phase 2 fine-tuning
        self.optim = 'Adam'

        self.iteration = 2000
        self.scheduler = False

        self.use_game = True
        self.earlystop = 1e-9
        self.save_final_img = False

        self.num_dummy = 1  # batch size
        self.num_exp = 1

        self.defense_method = 'none'  # 'none', 'soteria', 'noise', 'clipping', 'sparsification'
        self.noise_std = 0.0001  # for noise defense
        self.max_grad_norm_clipping = 4.0  # for clip defense
        self.sparsification_defense_sparsity = 90  # for sparsification_defense

        self.methods = ['HybridStealth']  # Use HybridStealth
        self.diff_task_agg = 'game'  # 'single', 'random', 'game'

        # === 核心参数修正 ===
        self.num_servers = 2  # 合谋服务器数量 (旧参数名)
        self.num_colluding = 2  # [关键] 必须添加，用于兼容 HybridStealth 代码

        self.num_clients = 100  # 参与安全聚合的客户端总数
        self.int_time = int(time.time())
        self.log_interval = 5
        self.log_metrics_interval = 100

        # === HybridStealth 所需特定参数 ===
        self.pmm_scale = 10.0  # [关键] PMM 损失缩放因子 [cite: 111]
        self.nash_lr = 0.1  # [关键] 修复报错: 纳什议价学习率
        self.tv_reg = 0.001  # [关键] TV 正则化权重 [cite: 319]
        self.covert_bits = 4  # [可选] 隐信道位数
        # =================================

        self.tv = 0.01
        self.eval_metrics = ['mse', 'lpips', 'psnr', 'ssim']

        self.train_data_loader_pickle_path = "data_loaders/cifar100/train_data_loader.pickle"
        self.test_data_loader_pickle_path = "data_loaders/cifar100/test_data_loader.pickle"

    # Getter 方法
    def get_num_clients(self):
        return self.num_clients

    def get_tv_reg(self):
        return self.tv_reg

    def get_logger(self):
        return self.logger

    def get_dataset(self):
        return self.dataset

    def get_eval_metrics(self):
        return self.eval_metrics

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path

    def get_num_dummy(self):
        return self.num_dummy

    def set_test_data_loader_pickle_path(self, path):
        self.test_data_loader_pickle_path = path

    def get_default_model_folder_path(self):
        return self.model_path

    def get_root_path(self):
        return self.root_path

    def get_debugOrRun(self):
        return self.debugOrRun

    def get_lr(self):
        return self.lr

    def get_earlystop(self):
        return self.earlystop

    def get_iteration(self):
        return self.iteration

    def get_num_exp(self):
        return self.num_exp

    def get_methods(self):
        return self.methods

    def get_start_index_str(self):
        return self.start_index_str

    def get_log_interval(self):
        return self.log_interval

    def get_net(self):
        return self.net

    def get_net_mt_diff(self):
        return self.net_mt_diff

    def get_imidx(self):
        return self.set_imidx

    def log(self):
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
            "Iteration: {}\n".format(self.iteration) + \
            "Learning Rate: {}\n".format(self.lr) + \
            "Model Path (Relative): {}\n".format(self.model_path) + \
            "Methods: {}\n".format(self.methods) + \
            "Number Exp: {}\n".format(self.num_exp) + \
            "Dataset: {}\n".format(self.dataset) + \
            "Number Server: {}\n".format(self.num_servers) + \
            "Number Client: {}\n".format(self.num_clients) + \
            "Set Imidx: {}\n".format(self.set_imidx) + \
            "Batch Size: {}\n".format(self.num_dummy) + \
            "Log Interval: {}\n".format(self.log_interval)