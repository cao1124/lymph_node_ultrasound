#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：classify_util.py
@IDE     ：PyCharm
@Author  ：cao xu
@Date    ：2025/6/19 上午10:19
"""

import os
import random
from enum import Enum
from PIL import ImageFilter
import cv2
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Sampler
from torchvision import transforms, models
from tqdm import tqdm
from warmup_scheduler import GradualWarmupScheduler


model_dict = {
    'efficientnet_b0': models.efficientnet_b0,
    'vgg16': models.vgg16,
    'vgg11': models.vgg11,
    "resnet18": models.resnet18,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
    "resnext50": models.resnext50_32x4d,
    "resnext101": models.resnext101_32x8d,
    "wide_resnet50_2": models.wide_resnet50_2,
    "wide_resnet101_2": models.wide_resnet101_2,
    'densenet121': models.densenet121,
    'densenet161': models.densenet161,
    'densenet169': models.densenet169,
    'densenet201': models.densenet201,
    "regnet_y_32gf": models.regnet_y_32gf,
}

mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
img_trans = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        # transforms.Grayscale(3),  # 将彩色图像转换为灰度图像，以鼓励网络学习到与颜色无关的特征。
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 0.3), value=0, inplace=False), # 会崩
        # Sharpen(p=0.5),  # 设置锐化的概率  # 增加锐化处理
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([224, 224]),
        transforms.Normalize(mass_mean, mass_std)
    ])
}


def func(listTemp, n, m=5):
    """ listTemp 为列表 平分后每份列表的的个数"""
    count = 0
    for i in range(0, len(listTemp), n):
        count += 1
        if count == m:
            yield listTemp[i:]
            break
        else:
            yield listTemp[i:i + n]


def read_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    return [line.strip().split() for line in lines]


def sec_pre(img, sec_model, device):
    with torch.no_grad():
        sec_output = sec_model(img.to(device))
        sec_pred = torch.softmax(sec_output, dim=1).cpu()
        sec_pred_cla = torch.argmax(sec_pred).numpy()
    return sec_pred_cla, sec_pred[0][sec_pred_cla].numpy()


def is_image(file_path):
    try:
        with Image.open(file_path) as img:
            img.verify()  # 验证文件是否为有效图像
        return True
    except (IOError, SyntaxError) as e:
        return False


def prepare_model(category_num, model_name, lr, num_epochs, device, weights):
    if model_name == 'convnext_large-1k':
        model = convnext_large(pretrained=True)
        model.head = nn.Linear(model.head.in_features, category_num)
    else:
        model = model_dict[model_name](pretrained=True)
        if model_name in ['resnet50', 'resnet101', 'resnet152', 'resnext50', 'wide_resnet50', 'resnext101',
                          'wide_resnet101']:
            model.fc = nn.Linear(in_features=2048, out_features=category_num, bias=True)
        elif model_name == 'resnet18':
            model.fc = nn.Linear(in_features=512, out_features=category_num, bias=True)
        elif model_name == 'densenet121':
            model.classifier = nn.Linear(in_features=1024, out_features=category_num, bias=True)
        elif model_name == 'densenet161':
            model.classifier = nn.Linear(in_features=2208, out_features=category_num, bias=True)
        elif model_name == 'densenet169':
            model.classifier = nn.Linear(in_features=1664, out_features=category_num, bias=True)
        elif model_name == 'densenet201':
            model.classifier = nn.Linear(in_features=1920, out_features=category_num, bias=True)
        elif model_name == 'regnet_y_32gf':
            model.fc = nn.Linear(in_features=3712, out_features=category_num, bias=True)
        elif model_name in ['vgg16', 'vgg11']:
            model.classifier[6] = nn.Linear(in_features=4096, out_features=category_num, bias=True)
        elif model_name in ['efficientnet_b0']:
            model.classifier[1] = nn.Linear(in_features=1280, out_features=category_num, bias=True)
    # 多GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    # 定义损失函数和优化器。
    # 定义loss权重  class_weights = torch.tensor([5.0, 1.0])     loss_func = nn.CrossEntropyLoss()
    loss_func = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)
    # optimizer = optim.NAdam(model.parameters(), lr=lr, betas=(0.8, 0.888), eps=1e-08, weight_decay=2e-4)
    # 定义学习率与轮数关系的函数
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)
    return model, optimizer, lr_scheduler, scheduler_warmup, loss_func


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        """
        Args:
            save_path : 模型保存文件夹
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)  # 这里会存储迄今最优模型的参数
        self.val_loss_min = val_loss


def cv_read(file_path, flag=-1):
    # 可读取图片（路径为中文）
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), flags=flag)
    # flag = -1,   8位深度，原通道
    # flag = 0，   8位深度，1通道
    # flag = 1，   8位深度，3通道
    # flag = 2，   原深度， 1通道
    # flag = 3，   原深度， 3通道
    # flag = 4，   8位深度，3通道
    return cv_img


def cv_write(file_path, file):
    cv2.imencode('.png', file)[1].tofile(file_path)


class BalanceDataSampler(Sampler):
    def __init__(self, dataset_targets, max_class=None):
        self.prob = []
        count = np.histogram(dataset_targets, len(set(dataset_targets)))[0]

        if max_class is None:
            max_class = count.max()

        modulos = max_class % count

        for key, y in enumerate(dataset_targets):
            self.prob += [key for i in range(max_class // count[y] + (modulos[y] > 0))]
            modulos[y] -= 1

    def __len__(self):
        return len(self.prob)

    def __iter__(self):
        return iter(np.array(self.prob)[np.random.permutation(len(self.prob))].tolist())


class DatasetTxt(Dataset):
    def __init__(self, img_folds, data_dir, transforms=None):
        self.transforms = transforms

        self.images = []
        self.names = []
        self.labels = []
        for fold in img_folds:
            for path in fold:
                img_path = data_dir + path.split(',')[0]
                cls = path.split(',')[1].replace('\n', '')
                img_name = img_path.split('/')[-1]
                self.images.append(img_path)
                self.names.append(img_name)
                # self.labels.append(LymphCls2CN[cls].value)
                self.labels.append(TransCls5CN[cls].value)
        self.length = len(self.images)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('RGB')
        if self.transforms is not None:
            try:
                img = self.transforms(img)
            except:
                print("Cannot transform image: {}".format(self.images[idx]))
        return img, self.labels[idx], self.names[idx]


def modelMap(model, device):
    if isinstance(model, nn.DataParallel):
        model = model.module
    return model.to(device)


class LymphCls2CN(Enum):
    转移淋巴瘤 = 0
    非转移淋巴瘤 = 1


class TransCls5CN(Enum):
    鼻咽 = 0
    肺 = 1
    乳腺 = 2
    食管 = 3
    其他 = 4