import matplotlib.pyplot as plt
import pickle
from loguru import logger
import arguments
import torchvision
import torch
import os
from torch.utils.data import Dataset
import PIL.Image as Image

def get_lfw_gender_mapping(lfw_path):
    images_all = []
    labels_all = []
    male_images = []
    female_images = []
    gender_mapping_dict = {0:[], 1:[]}
    folders = os.listdir(lfw_path)
    for foldidx, fold in enumerate(folders):
        files = os.listdir(os.path.join(lfw_path, fold))
        for f in files:
            if len(f) > 4 and f[-4:] == '.jpg':
                images_all.append(os.path.join(lfw_path, fold, f))
                labels_all.append(foldidx)
    with open('male_names.txt', 'r') as f:
        for line in f:
            male_images.append(line.strip('\n').split(',')[0])

    with open('female_names.txt', 'r') as f:
        for line in f:
            female_images.append(line.strip('\n').split(',')[0])

    for i in range(len(images_all)):
        if images_all[i].split('/')[-1] in male_images:
            gender_mapping_dict[0].append(labels_all[i])
        elif images_all[i].split('/')[-1] in female_images:
            gender_mapping_dict[1].append(labels_all[i])
        else:
            gender_mapping_dict[0].append(labels_all[i])
    gender_mapping_dict[0] = list(set(gender_mapping_dict[0]))
    gender_mapping_dict[1] = list(set(gender_mapping_dict[1]))

    return gender_mapping_dict

def weights_init(m):
    try:
        if hasattr(m, "weight"):
            m.weight.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.weight' % m._get_name())
    try:
        if hasattr(m, "bias"):
            m.bias.data.uniform_(-0.5, 0.5)
    except Exception:
        print('warning: failed in weights_init for %s.bias' % m._get_name())

def set_idx(imidx, imidx_list, idx_shuffle):

    '''choose set image or random'''
    args = arguments.Arguments(logger)
    if args.get_imidx() == 000000:
        idx = idx_shuffle[imidx]
        imidx_list.append(idx)
    else:
        idx = args.get_imidx()
        imidx_list.append(idx)

    return  idx, imidx_list


def label_mapping(origin_label, idx):
    args = arguments.Arguments(logger)
    mapping_dict = {}
    if args.get_dataset() == 'mnist' or args.get_dataset() == 'stl10':
        if origin_label < 5:
            tmp_label_1 = torch.Tensor([0]).long()
        else:
            tmp_label_1 = torch.Tensor([1]).long()

    elif args.get_dataset() == 'cifar100':
        if idx == 0:
            mapping_dict = {0: [4, 30, 55, 72, 95],
                            1: [1, 32, 67, 73, 91],
                            2: [54, 62, 70, 82, 92],
                            3: [9, 10, 16, 28, 61],
                            4: [0, 51, 53, 57, 83],
                            5: [22, 39, 40, 86, 87],
                            6: [5, 20, 25, 84, 94],
                            7: [6, 7, 14, 18, 24],
                            8: [3, 42, 43, 88, 97],
                            9: [12, 17, 37, 68, 76],
                            10: [23, 33, 49, 60, 71],
                            11: [15, 19, 21, 31, 38],
                            12: [34, 63, 64, 66, 75],
                            13: [26, 45, 77, 79, 99],
                            14: [2, 11, 35, 46, 98],
                            15: [27, 29, 44, 78, 93],
                            16: [36, 50, 65, 74, 80],
                            17: [47, 52, 56, 59, 96],
                            18: [8, 13, 48, 58, 90],
                            19: [41, 69, 81, 85, 89]}
        elif idx == 1:
            mapping_dict = {0: [4, 30, 55, 72, 95, 1, 32, 67, 73, 91],
                            1: [54, 62, 70, 82, 92, 9, 10, 16, 28, 61],
                            2: [0, 51, 53, 57, 83,22, 39, 40, 86, 87 ],
                            3: [5, 20, 25, 84, 94, 6, 7, 14, 18, 24],
                            4: [3, 42, 43, 88, 97, 12, 17, 37, 68, 76],
                            5: [23, 33, 49, 60, 71, 15, 19, 21, 31, 38],
                            6: [34, 63, 64, 66, 75, 26, 45, 77, 79, 99],
                            7: [2, 11, 35, 46, 98, 27, 29, 44, 78, 93],
                            8: [36, 50, 65, 74, 80,47, 52, 56, 59, 96 ],
                            9: [8, 13, 48, 58, 90, 41, 69, 81, 85, 89],}
        elif idx == 2:
            mapping_dict = {0: [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 28, 61],
                            1: [0, 51, 53, 57, 83,22, 39, 40, 86, 87 , 5, 20, 25, 84, 94, 6, 7, 14, 18, 24],
                            2: [3, 42, 43, 88, 97, 12, 17, 37, 68, 76, 23, 33, 49, 60, 71, 15, 19, 21, 31, 38],
                            3: [34, 63, 64, 66, 75, 26, 45, 77, 79, 99, 2, 11, 35, 46, 98, 27, 29, 44, 78, 93],
                            4: [36, 50, 65, 74, 80,47, 52, 56, 59, 96 , 8, 13, 48, 58, 90, 41, 69, 81, 85, 89],}

        elif idx == 3:
            mapping_dict = {0: [4, 30, 55, 72, 95, 1, 32, 67, 73, 91, 54, 62, 70, 82, 92, 9, 10, 16, 28, 61, 0, 51, 53, 57, 83,22, 39, 40, 86, 87 , 5, 20, 25, 84, 94, 6, 7, 14, 18, 24, 3, 42, 43, 88, 97, 12, 17, 37, 68, 76,],
                            1: [ 23, 33, 49, 60, 71, 15, 19, 21, 31, 38, 34, 63, 64, 66, 75, 26, 45, 77, 79, 99, 2, 11, 35, 46, 98, 27, 29, 44, 78, 93, 36, 50, 65, 74, 80,47, 52, 56, 59, 96 , 8, 13, 48, 58, 90, 41, 69, 81, 85, 89],}

        tmp_label_1 = torch.Tensor([k for k, v in mapping_dict.items() if origin_label in v]).long()

    elif args.get_dataset() == 'stl10':
        mapping_dict = {0: [0, 2, 8, 9],
                        1: [1, 3, 4, 5, 6, 7],}

        tmp_label_1 = torch.Tensor([k for k, v in mapping_dict.items() if origin_label in v]).long()



    elif args.get_dataset() == 'lfw':
        mapping_dict = get_lfw_gender_mapping('./data/lfw')
        # print(mapping_dict)
        tmp_label_1 = torch.Tensor([k for k, v in mapping_dict.items() if origin_label in v]).long()

    else:
        print('no other label found')
        tmp_label_1 = torch.Tensor([origin_label]).long()

    return tmp_label_1


class Dataset_from_Image(Dataset):
    def __init__(self, imgs, labs, transform=None):
        self.imgs = imgs  # img paths
        self.labs = labs  # labs is ndarray
        self.transform = transform
        del imgs, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        lab = self.labs[idx]
        img = Image.open(self.imgs[idx])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.transform(img)
        return img, lab



def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
    return dx + dy
