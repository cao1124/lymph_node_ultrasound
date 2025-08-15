#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：multitask_util.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/8/15 下午2:20 
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


# ================== 新增多任务模型 ==================
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
            # backbone 的分类层不可用（比如已经被替换成 Identity），用一个 deepcopy 在 cpu 上做一次前向推理来推断输出维度
            tmp = copy.deepcopy(backbone).cpu()
            tmp.eval()
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                out = tmp(dummy)
                # 如果输出是 feature map（4D），先 flatten
                if out.dim() > 2:
                    out = out.view(out.size(0), -1)
                in_features = out.shape[1]

        # 添加多头分类器
        self.head_main = nn.Linear(in_features, num_classes_main)
        self.head_aux = nn.Linear(in_features, num_classes_aux)

    def forward(self, x, return_features=False):
        """
        x: tensor, shape (B, C, H, W)
        return_features: 如果为 True，则额外返回 backbone 提取到的特征向量
        返回: (outputs_main, outputs_aux) 或 (outputs_main, outputs_aux, features)
        """
        features = self.backbone(x)

        # backbone 有时会返回 feature map (B, C, H, W)，有时直接 (B, F)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)

        out_main = self.head_main(features)
        out_aux = self.head_aux(features)

        if return_features:
            return out_main, out_aux, features
        return out_main, out_aux


# ================== 新增Mean Teacher模型 ==================
class MeanTeacherModel(nn.Module):
    def __init__(self, student_model):
        super(MeanTeacherModel, self).__init__()
        self.student = student_model
        self.teacher = self._create_teacher(student_model)

    def _create_teacher(self, student_model):
        # 如果 student_model 是 DataParallel，取出 module（真实模型）
        is_dp = isinstance(student_model, nn.DataParallel)
        student_core = student_model.module if is_dp else student_model
        # 直接 deepcopy 整个 student_core（保留结构和参数）
        teacher_core = copy.deepcopy(student_core)
        # 把 teacher_core 包装回 DataParallel（如果原 student 是 DataParallel）
        if is_dp:
            teacher = nn.DataParallel(teacher_core)
        else:
            teacher = teacher_core
        # 使 teacher 参数不参与梯度计算
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def update_teacher(self, alpha=0.99):
        # EMA更新teacher参数
        with torch.no_grad():
            for param_s, param_t in zip(self.student.parameters(), self.teacher.parameters()):
                param_t.data = alpha * param_t.data + (1 - alpha) * param_s.data

    def forward(self, x, is_teacher=False, return_features=False):
        if is_teacher:
            return self.teacher(x, return_features)
        return self.student(x, return_features)


# ================== 新增数据增强 ==================
class GaussianBlur:
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# ================== 增强的数据预处理 ==================
mass_mean, mass_std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
img_trans = {
    'train': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomGrayscale(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)]),

    'strong': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.RandomGrayscale(p=0.8),
        transforms.RandomHorizontalFlip(p=0.8),
        transforms.RandomVerticalFlip(p=0.8),
        transforms.RandomRotation(120),
        transforms.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.5),
        transforms.GaussianBlur(kernel_size=(7, 11), sigma=(0.1, 3.0)),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)]),

    'valid': transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mass_mean, mass_std)
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
        """
        重构后的数据集类，支持不同类型标签
        Args:
            data_list: 数据列表
            data_dir: 图像基础路径
            transform: 基础数据增强
            strong_transform: 强数据增强
            has_main_label: 是否包含主任务标签（良恶性）
            has_aux_label: 是否包含辅任务标签（肿大状态）
        """
        self.data_dir = data_dir
        self.transform = transform
        self.strong_transform = strong_transform
        self.has_main_label = has_main_label
        self.has_aux_label = has_aux_label

        self.images = []
        self.labels_main = [] if has_main_label else None
        self.labels_aux = [] if has_aux_label else None
        self.names = []

        # 解析数据列表
        for item in data_list:
            parts = item.strip().split(',')
            img_path = os.path.join(data_dir, parts[0])

            if not os.path.exists(img_path):
                print(f"Warning: Image not found - {img_path}")
                continue

            self.images.append(img_path)
            self.names.append(os.path.basename(img_path))

            # 解析标签
            if has_main_label and len(parts) > 1:
                self.labels_main.append(1 if parts[1] == '恶性' else 0)   # 良恶性分类标签（恶性=1，良性=0）
            elif has_main_label:
                self.labels_main.append(-1)  # 缺失标签

            if has_aux_label and len(parts) > (2 if has_main_label else 1):
                idx = 2 if has_main_label else 1
                self.labels_aux.append(1 if parts[idx] == '肿大' else 0)   # 肿大状态标签（肿大=1，非肿大=0）
            elif has_aux_label:
                self.labels_aux.append(-1)  # 缺失标签

        print(f"Loaded {len(self.images)} samples - Main Label: {has_main_label}, Aux Label: {has_aux_label}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert('RGB')

        # 基础变换
        img_base = self.transform(img) if self.transform else img

        # 强数据增强
        img_strong = self.strong_transform(img) if self.strong_transform else img_base

        # 标签处理
        label_main = self.labels_main[idx] if self.has_main_label else -1
        label_aux = self.labels_aux[idx] if self.has_aux_label else -1

        return {
            'image': img_base,
            'image_strong': img_strong,
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
            self.prob += [key for i in range(max_class // count[y] + (modulos[y] > 0))]
            modulos[y] -= 1

    def __len__(self):
        return len(self.prob)

    def __iter__(self):
        return iter(np.array(self.prob)[np.random.permutation(len(self.prob))].tolist())


# ================== 模型准备函数 ==================
def prepare_multi_task_model(category_num_main, category_num_aux, model_name, lr, num_epochs, device,
                             weights_main=None):
    # 创建基础模型
    base_model = model_dict[model_name](pretrained=True)

    # 创建多任务模型
    model = MultiTaskModel(base_model, category_num_main, category_num_aux)

    # 多GPU支持
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model.to(device)

    # 损失函数
    loss_func_main = nn.CrossEntropyLoss(weight=weights_main.to(device) if weights_main is not None else None)
    loss_func_aux = nn.CrossEntropyLoss()  # 肿大状态分类任务通常类别均衡

    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)
    lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=lr * 0.01)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)

    return model, optimizer, lr_scheduler, scheduler_warmup, loss_func_main, loss_func_aux


# ================== 早停机制 ==================
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
