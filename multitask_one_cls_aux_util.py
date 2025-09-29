#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：multitask_one_cls_aux_util.py
@IDE     ：PyCharm
@Author  ：cao xu (edited)
@Date    ：2025/9/29

说明：
- 辅任务 head 改为嵌入头（输出 z_aux），用于一类学习（只用“肿大”正样本）。
- 新增 MetastasisCls2CN（主任务：未转移/转移）。
- 数据集解析支持更通用的 0/1/True/False/中英文关键字映射。
- prepare_multi_task_model 去掉 num_classes_aux；返回不再含 loss_func_aux。
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


class MultiTaskModel(nn.Module):
    def __init__(self, backbone, num_classes_main: int, aux_embed_dim: int = 128):
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

        # 主任务：转移/未转移 二分类
        self.head_main = nn.Linear(in_features, num_classes_main)

        # 辅任务：一类嵌入（只用肿大正样本），不做 softmax
        self.head_aux = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_features // 2, aux_embed_dim)
        )

    def forward(self, x, return_features=False):
        feats = self.backbone(x)
        if feats.dim() > 2:
            feats = feats.view(feats.size(0), -1)
        out_main = self.head_main(feats)
        z_aux = self.head_aux(feats)
        if return_features:
            return out_main, z_aux, feats
        return out_main, z_aux


class MeanTeacherModel(nn.Module):
    def __init__(self, student_model: nn.Module):
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


class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]

# 弱/强增强策略
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

# ================== 更通用的标签解析 ==================
_POS = {
    '1', 'true', 'yes', 'y', 't', 'pos', 'positive',
    '恶性', '转移', '肿大', '是', '阳性', 'malignant', 'metastasis', 'swollen'
}
_NEG = {
    '0', 'false', 'no', 'n', 'f', 'neg', 'negative',
    '良性', '未转移', '非转移', '非肿大', '不肿大', '否', '阴性', 'benign', 'non-metastasis', 'not-swollen'
}

def parse_binary_label(val: str) -> int:
    if val is None:
        return -1
    s = str(val).strip().lower()
    if s in _POS:
        return 1
    if s in _NEG:
        return 0
    # 尝试整数化
    try:
        iv = int(s)
        if iv in (0, 1):
            return iv
    except Exception:
        pass
    return -1


class MetastasisCls2CN(Enum):
    非转移淋巴瘤 = 0
    转移淋巴瘤 = 1

class SwollenStatus(Enum):
    非肿大 = 0
    肿大 = 1


class WeaklyLabeledDataset(Dataset):
    """通用列表格式：
    每行： path[, main_label][, aux_label]
    - 对于主任务数据：has_main_label=True, has_aux_label=False
    - 对于辅任务数据（只用肿大正样本）：has_main_label=False, has_aux_label=True（可无标签字段）
    """
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
                    self.labels_main.append(parse_binary_label(parts[1]))
                else:
                    self.labels_main.append(-1)

            if has_aux_label:
                idx = 2 if has_main_label else 1
                if len(parts) > idx:
                    self.labels_aux.append(parse_binary_label(parts[idx]))
                else:
                    # 对于只用“肿大正样本”的场景，标签可缺省；这里置为1或-1都不影响训练
                    self.labels_aux.append(1)

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
            'image': img_weak,          # 弱增强（主任务 & 辅任务都可用）
            'image_strong': img_strong, # 强增强
            'label_main': label_main,
            'label_aux': label_aux,
            'name': self.names[idx]
        }


class BalanceDataSampler(Sampler):
    def __init__(self, dataset_targets, max_class=None):
        # 过滤 -1（未标注）的情形
        dataset_targets = [t for t in dataset_targets if t in (0, 1)]
        self.prob = []
        if len(dataset_targets) == 0:
            return
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
        if len(self.prob) == 0:
            return iter([])
        return iter(np.array(self.prob)[np.random.permutation(len(self.prob))].tolist())


def prepare_multi_task_model(category_num_main, model_name, lr, num_epochs, device,
                             aux_embed_dim: int = 128, weights_main=None):
    base_model = model_dict[model_name](pretrained=True)
    model = MultiTaskModel(base_model, category_num_main, aux_embed_dim)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func_main = nn.CrossEntropyLoss(
        weight=weights_main.to(device) if weights_main is not None else None,
        label_smoothing=0.05
    )

    # 优化器：分组LR + 更强正则
    backbone_params, head_params = [], []
    for n, p in model.named_parameters():
        if 'head_main' in n or 'head_aux' in n:
            head_params.append(p)
        else:
            backbone_params.append(p)

    optimizer = optim.SGD([
        {'params': backbone_params, 'lr': lr * 0.5},
        {'params': head_params,     'lr': lr * 1.0}
    ], weight_decay=5e-4, momentum=0.9, nesterov=True)

    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=cosine)

    # 返回：不再包含 loss_func_aux
    return model, optimizer, cosine, scheduler_warmup, loss_func_main


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