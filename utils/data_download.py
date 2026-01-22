import torch
import torchvision
from arguments import Arguments
from loguru import logger
import os
import pathlib
import sys
from torchvision import datasets, transforms
import numpy as np
from .data_processing import Dataset_from_Image



def lfw_dataset(lfw_path, shape_img):
    images_all = []
    labels_all = []
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    transform = transforms.Compose([transforms.Resize(size=shape_img)])
    dst = Dataset_from_Image(images_all, np.asarray(labels_all, dtype=int), transform=transform)
    return dst

def load_data(dataset, root_path, data_path, save_path):


    tt = transforms.Compose([transforms.ToTensor()])
    tp = transforms.Compose([transforms.ToPILImage()])

    d_mean = [0.5, 0.5, 0.5]
    d_std = [0.5, 0.5, 0.5]

    if not os.path.exists('results'):
        os.mkdir('results')
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    ''' load data '''
    if dataset == 'mnist':
        shape_img = (28, 28)
        input_size = 28
        num_classes = 10
        alter_num_classes = 2
        channel = 1
        hidden = 588
        dst = datasets.MNIST(data_path, download=True)

        d_mean = [0.5]
        d_std = [0.5]

    elif dataset == 'cifar100':
        shape_img = (32, 32)
        input_size = 32
        num_classes = 100
        alter_num_classes = 20
        channel = 3
        hidden = 768
        dst = datasets.CIFAR100(data_path, download=True)

        d_mean = [0.5071598291397095, 0.4866936206817627, 0.44120192527770996]
        d_std = [0.2673342823982239, 0.2564384639263153, 0.2761504650115967]
        tt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std)])

    elif dataset == 'cifar10':
        shape_img = (32, 32)
        input_size = 32
        num_classes = 10
        alter_num_classes = 20
        channel = 3
        hidden = 768
        dst = datasets.CIFAR10(data_path, download=True)

        d_mean = [0.4914672374725342, 0.4822617471218109, 0.4467701315879822]
        d_std = [0.24703224003314972, 0.24348513782024384, 0.26158785820007324]
        tt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(d_mean, d_std)])

    elif dataset == 'stl10':

        shape_img = (96,96)
        input_size = 96
        num_classes = 10
        alter_num_classes = 2
        channel = 3
        hidden = 6912
        dst = datasets.STL10(data_path, download=True)

        tt = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([32,32]),
            transforms.Normalize(d_mean, d_std)])

    elif dataset == 'lfw':
        shape_img = (32, 32)
        input_size = 32
        num_classes = 5749
        alter_num_classes = 2
        channel = 3
        hidden = 768
        lfw_path = os.path.join(root_path, 'data/lfw')
        dst = lfw_dataset(lfw_path, shape_img)

    else:
        exit('unknown dataset')

    idx_shuffle = np.random.permutation(len(dst))

    return tt, tp, num_classes, alter_num_classes, channel, hidden, dst, input_size, idx_shuffle, [d_mean, d_std]
