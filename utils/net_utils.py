from utils import FC2, LeNet, MNISTCNN, Cifar100ResNet
from utils.data_processing import weights_init
import torch.nn.init as init
import torch.nn as nn
import torch


def init_params(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            init.kaiming_normal_(m.weight, mode='fan_in')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=1e-3)
            if m.bias is not None:
                init.constant_(m.bias, 0)


def intialize_nets(args, method, channel, hidden, num_classes,alter_num_classes, input_size):
    SEED = 42

    def set_random_seed(seed=233):
        import random
        import numpy as np
        """233 = 144 + 89 is my favorite number."""
        torch.manual_seed(seed + 1)
        torch.cuda.manual_seed(seed + 2)
        torch.cuda.manual_seed_all(seed + 3)
        np.random.seed(seed + 4)
        torch.cuda.manual_seed_all(seed + 5)
        random.seed(seed + 6)

    set_random_seed(SEED)

    nets = []
    if method == 'mDLG_mt':
        args.logger.info('running different task multi server')
        if args.get_dataset() == 'mnist':
            args.logger.info('running same structure multiserver mnist')
            class_list  = [num_classes, alter_num_classes]
            for i in range(args.num_servers):
                net = LeNet(channel=channel, hidden=hidden, num_classes= class_list[i])
                net.apply(weights_init)
                nets.append(net)
        elif args.get_dataset() == 'stl10':
            if args.net == 'resnet34':
                from utils.resnet.ResNet import resnet34 as resnet
            else:  # args.net == 'resnet20-4'
                from utils.resnet.ResNet import resnet20_4 as resnet
            args.logger.info('running same structure multiserver stl10')
            class_list = [num_classes, alter_num_classes]
            for i in range(args.num_servers):
                if i == 0:
                    net = resnet(num_classes=class_list[i], no_init=True)
                else:
                    net = resnet(num_classes=class_list[i], no_init=False)
                nets.append(net)
        elif args.get_dataset() == 'cifar100':
            args.logger.info('running same structure multiserver cifar100')
            class_list  = [num_classes, alter_num_classes, 10, 5, 2]
            if args.net == 'lenet':
                for i in range(args.num_servers):
                    net = LeNet(channel=channel, hidden=hidden, num_classes= class_list[i])
                    net.apply(weights_init)
                    nets.append(net)
            else: # 'resnet20-4'
                from utils.resnet.ResNet import resnet20_4
                for i in range(args.num_servers):
                    if i == 0:
                        net = resnet20_4(num_classes=class_list[i], no_init=True)
                    else:
                        net = resnet20_4(num_classes=class_list[i], no_init=False)
                    nets.append(net)

        elif args.get_dataset() == 'lfw':
            args.logger.info('running same structure multiserver lfw')
            class_list  = [num_classes, alter_num_classes]
            for i in range(args.num_servers):
                net = LeNet(channel=channel, hidden=hidden, num_classes= class_list[i])
                net.apply(weights_init)
                nets.append(net)

        else:
            class_list  = [num_classes, alter_num_classes,]
            for i in range(args.num_servers):
                net = LeNet(channel=channel, hidden=hidden, num_classes= class_list[0])
                net.apply(weights_init)
                nets.append(net)

    elif method == 'mDLG':
        args.logger.info('running same task multi server')
        num_servers = args.num_servers
        args.logger.info('number of servers: #{}', num_servers)
        if args.get_net() == "lenet":
            for i in range(num_servers):
                net = LeNet(channel=channel, hidden=hidden,num_classes = num_classes)
                # net.apply(weights_init)
                nets.append(net)
        elif args.get_net() == 'vgg11':
            from utils.vgg import VGG11
            for i in range(num_servers):
                net = VGG11()
                nets.append(net)
        elif args.get_net() == 'resnet':
            for i in range(num_servers):
                net = Cifar100ResNet(num_classes=num_classes)
                init_params(net)
                nets.append(net)
        elif args.get_net() == 'fc2':
            for i in range(num_servers):
                net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
                net.apply(weights_init)
                nets.append(net)
        elif args.get_net() == 'resnet20-4':
            from utils.resnet.ResNet import resnet20_4
            for i in range(num_servers):
                net = resnet20_4(num_classes=num_classes)
                nets.append(net)
        elif args.get_net() == 'resnet34':
            from utils.resnet.ResNet import resnet34
            for i in range(num_servers):
                net = resnet34(num_classes=num_classes)
                nets.append(net)
        elif args.get_net() == 'resnet50':
            from utils.resnet.ResNet import resnet50
            for i in range(num_servers):
                net = resnet50(num_classes=num_classes)
                nets.append(net)
    else:
        args.logger.info('running single server')
        for i in range(args.num_servers):
            if args.get_net() == 'lenet':
                net = LeNet(channel=channel, hidden=hidden, num_classes=num_classes)
                net.apply(weights_init)
                nets.append(net)
            elif args.get_net() == 'vgg11':
                from utils.vgg import VGG11
                net = VGG11()
                nets.append(net)
            elif args.get_net() == 'fc2':
                net = FC2(channel=channel, input_size=input_size, hidden=500, num_classes=num_classes)
            # net.apply(weights_init)
                nets.append(net)
            elif args.get_net() == 'resnet':
                net = Cifar100ResNet(num_classes = num_classes)
                net.apply(weights_init)
                nets.append(net)
            elif args.get_net() == 'resnet20-4':
                from utils.resnet.ResNet import resnet20_4
                net = resnet20_4(num_classes=num_classes)
                nets.append(net)
            elif args.get_net() == 'resnet34':
                from utils.resnet.ResNet import resnet34
                net = resnet34(num_classes=num_classes)
                nets.append(net)
            elif args.get_net() == 'resnet50':
                from utils.resnet.ResNet import resnet50
                net = resnet50(num_classes=num_classes)
                nets.append(net)

    for net in nets:
        net.eval()

    return nets

