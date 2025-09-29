#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：multitask_util.py
@IDE     ：PyCharm
@Author  ：cao xu (edited)
@Date    ：2025/9/10

说明：
- 保留你的多任务结构与均值教师结构
- 调整数据增强（医学友好）
- 其余工具函数保持接口不变
"""
import copy
import os
import random
from enum import Enum
from PIL import ImageFilter
from PIL import Image
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Sampler
from torchvision import transforms, models
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

# ================== 多任务模型 ==================
class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_classes_main, num_classes_aux):
        super(MultiTaskModel, self).__init__()
        self.backbone = backbone

        # 更鲁棒地获取 in_features
        if hasattr(backbone, 'fc') and hasattr(backbone.fc, 'in_features'):
            in_features = backbone.fc.in_features
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier') and hasattr(backbone.classifier, 'in_features'):
            in_features = backbone.classifier.in_features
            backbone.classifier = nn.Identity()
        else:
            tmp = copy.deepcopy(backbone).cpu()
            tmp.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = tmp(dummy)
                if out.dim() > 2:
                    out = out.view(out.size(0), -1)
                in_features = out.shape[1]

        self.head_main = nn.Linear(in_features, num_classes_main)
        self.head_aux = nn.Linear(in_features, num_classes_aux)

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        out_main = self.head_main(feats)
        out_aux = self.head_aux(feats)
        if return_features:
            return out_main, out_aux, feats
        return out_main, out_aux


# ================== Mean Teacher ==================
class MeanTeacherModel(nn.Module):
    def __init__(self, student_model):
        super(MeanTeacherModel, self).__init__()
        self.student = student_model
        self.teacher = self._create_teacher(student_model)

    def _create_teacher(self, student_model):
        is_dp = isinstance(student_model, nn.DataParallel)
        student_core = student_model.module if is_dp else student_model
        teacher_core = copy.deepcopy(student_core)
        teacher = nn.DataParallel(teacher_core) if is_dp else teacher_core
        for p in teacher.parameters():
            p.requires_grad = False
        return teacher

    def update_teacher(self, alpha=0.99):
        with torch.no_grad():
            for ps, pt in zip(self.student.parameters(), self.teacher.parameters()):
                pt.data = alpha * pt.data + (1 - alpha) * ps.data

    def forward(self, x, is_teacher=False, return_features=False):
        if is_teacher:
            return self.teacher(x, return_features)
        return self.student(x, return_features)


# ================== 数据增强（更医学友好） ==================
class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# 你原先的均值方差保留
mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]

# FIX: 调整增强策略（弱增强保守；强增强不改变色调，仅亮度/对比度）
img_trans = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std),
    ]),
    'strong': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.4, contrast=0.4),  # 不改 hue/sat
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std),
    ]),
    'valid': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std),
    ])
}

# ================== 枚举定义 ==================
class LymphPathologicCls2CN(Enum):
    良性 = 0
    恶性 = 1

class SwollenStatus(Enum):
    非肿大 = 0
    肿大 = 1

# ================== 数据集类 ==================
class WeaklyLabeledDataset(Dataset):
    def __init__(self, data_list, data_dir, transform=None, strong_transform=None,
                 has_main_label=False, has_aux_label=False):
        self.data_dir = data_dir
        self.transform = transform
        self.strong_transform = strong_transform
        self.has_main_label = has_main_label
        self.has_aux_label = has_aux_label

        self.images, self.names = [], []
        self.labels_main = [] if has_main_label else None
        self.labels_aux = [] if has_aux_label else None

        for item in data_list:
            parts = item.strip().split(',')
            img_path = os.path.join(data_dir, parts[0])
            if not os.path.exists(img_path):
                print(f"Warning: Image not found - {img_path}")
                continue

            self.images.append(img_path)
            self.names.append(os.path.basename(img_path))

            if has_main_label:
                if len(parts) > 1:
                    self.labels_main.append(1 if parts[1] == '恶性' else 0)
                else:
                    self.labels_main.append(-1)

            if has_aux_label:
                idx = 2 if has_main_label else 1
                if len(parts) > idx:
                    self.labels_aux.append(1 if parts[idx] == '肿大' else 0)
                else:
                    self.labels_aux.append(-1)

        print(f"Loaded {len(self.images)} samples - Main Label: {has_main_label}, Aux Label: {has_aux_label}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        img_weak = self.transform(img) if self.transform else img
        img_strong = self.strong_transform(img) if self.strong_transform else img_weak

        label_main = self.labels_main[idx] if self.has_main_label else -1
        label_aux = self.labels_aux[idx] if self.has_aux_label else -1

        return {
            'image': img_weak,         # 弱增强
            'image_strong': img_strong,  # 强增强
            'label_main': label_main,
            'label_aux': label_aux,
            'name': self.names[idx]
        }

# ================== 平衡采样器 ==================
class BalanceDataSampler(Sampler):
    def __init__(self, dataset_targets, max_class=None):
        self.prob = []
        count = np.histogram(dataset_targets, len(set(dataset_targets)))[0]
        if max_class is None:
            max_class = count.max()
        modulos = max_class % count
        for key, y in enumerate(dataset_targets):
            self.prob += [key for _ in range(max_class // count[y] + (modulos[y] > 0))]
            modulos[y] -= 1

    def __len__(self):
        return len(self.prob)

    def __iter__(self):
        return iter(np.array(self.prob)[np.random.permutation(len(self.prob))].tolist())

# ================== 模型准备 ==================
def prepare_multi_task_model(category_num_main, category_num_aux, model_name, lr, num_epochs, device,
                             weights_main=None):
    base_model = model_dict[model_name](pretrained=True)
    model = MultiTaskModel(base_model, category_num_main, category_num_aux)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func_main = nn.CrossEntropyLoss(weight=weights_main.to(device) if weights_main is not None else None, label_smoothing=0.05)
    loss_func_aux = nn.CrossEntropyLoss()

    # optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)
    # 优化器：分组LR + 更强正则
    backbone_params = []
    head_params = []
    for n, p in model.named_parameters():
        if 'head_main' in n or 'head_aux' in n:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = optim.SGD([{'params': backbone_params, 'lr': lr * 0.5}, {'params': head_params,     'lr': lr * 1.0}],
                          weight_decay=5e-4, momentum=0.9, nesterov=True)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=cosine)

    # NOTE: 返回仅 warmup 调用即可（不要再单独 scheduler.step()）
    return model, optimizer, cosine, scheduler_warmup, loss_func_main, loss_func_aux

# ================== 早停 ==================
class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
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
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        path = os.path.join(self.save_path, 'best_network.pth')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
