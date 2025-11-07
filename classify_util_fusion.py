#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Fusion utilities for dual-image classification (original + ROI crop).
Backbone: ResNet-50 by default. Fusion: feature concatenation -> classifier.
This module is designed to plug into the training loop style in classify_txt.py.
"""
import os
from typing import List, Tuple
from enum import Enum

import torch
from torch import nn, optim
from torch.utils.data import Dataset
from torchvision import transforms, models
from PIL import Image

# Reuse enums and transforms from the user's original utilities if present.
# If needed, you can fall back to the simple defaults below.
try:
    from classify_util import img_trans, TransCls6CN  # type: ignore
except Exception:
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
        'valid': transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([224, 224]),
            transforms.Normalize(mass_mean, mass_std)
        ])
    }


class TransCls6CN(Enum):
    肺癌 = 0
    乳腺癌 = 1
    食管癌 = 2
    鼻咽癌 = 3
    腹腔肿瘤 = 4
    其他 = 5


class DatasetTxtTwo(Dataset):
    """
    Read a single txt that lists ORIGINAL image path and label, one per line:
        relative/or/absolute/path/to/image.jpg,<中文标签名>
    This dataset loads two images per sample:
        - big image: the original path
        - crop image: the path after applying the fixed replace rule:
              replace('20251105-最后测试内外部人机对比', '20251105-最后测试内外部人机对比-crop')
    It returns ((img_big, img_crop), label_index, image_name).
    """
    def __init__(self, img_folds: List[List[str]], data_dir: str, transforms=None):
        self.transforms = transforms
        self.images_big: List[str] = []
        self.images_crop: List[str] = []
        self.names: List[str] = []
        self.labels: List[int] = []

        for fold in img_folds:
            for line in fold:
                raw_path = line.split(',')[0].strip()
                cls_cn = line.split(',')[1].strip()
                big_path = os.path.join(data_dir, raw_path) if data_dir else raw_path
                crop_path = big_path.replace('20251105-最后测试内外部人机对比',
                                             '20251105-最后测试内外部人机对比-crop')

                img_name = os.path.basename(big_path)
                self.images_big.append(big_path)
                self.images_crop.append(crop_path)
                self.names.append(img_name)
                # map chinese class name to index using user's enum (TransCls6CN)
                self.labels.append(TransCls6CN[cls_cn].value)

        self.length = len(self.images_big)

    def __len__(self):
        return self.length

    def _load(self, path: str):
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __getitem__(self, idx: int):
        big = self._load(self.images_big[idx])
        crop = self._load(self.images_crop[idx])
        return (big, crop), self.labels[idx], self.names[idx]


class FusionResNet(nn.Module):
    """
    Two-branch ResNet backbone (one for original, one for crop).
    Extract features from each branch (2048-d), fuse by concatenation (4096-d),
    then classify.
    """
    def __init__(self, num_classes: int, backbone: str = "resnet50"):
        super().__init__()
        # Build feature extractors
        self.backbone_name = backbone
        self.branch_big = self._make_backbone(backbone)
        self.branch_crop = self._make_backbone(backbone)

        # Feature dims per ResNet variant
        feat_dim = 2048 if backbone in ["resnet50", "resnet101", "resnet152",
                                        "resnext50_32x4d", "resnext101_32x8d",
                                        "wide_resnet50_2", "wide_resnet101_2"] else 512

        fused_dim = feat_dim * 2
        self.fusion = nn.Sequential(
            nn.BatchNorm1d(fused_dim),
            nn.Dropout(p=0.3),
            nn.Linear(fused_dim, 1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Dropout(p=0.3),
            nn.Linear(1024, num_classes)
        )

    def _make_backbone(self, name: str):
        # Create a torchvision model and strip the classifier to get global features.
        if name == "resnet18":
            net = models.resnet18(pretrained=True)
            feat_dim = 512
        elif name == "resnet50":
            net = models.resnet50(pretrained=True)
            feat_dim = 2048
        elif name == "resnet101":
            net = models.resnet101(pretrained=True)
            feat_dim = 2048
        else:
            # default to resnet50
            net = models.resnet50(pretrained=True)
            feat_dim = 2048

        # Replace the last FC layer by identity to output pooled features.
        net.fc = nn.Identity()
        self._feat_dim = feat_dim
        return net

    def forward(self, x):
        # x is expected to be a tuple: (big_batch, crop_batch)
        if not isinstance(x, (tuple, list)) or len(x) != 2:
            raise ValueError("FusionResNet expects a tuple (big_batch, crop_batch) as input.")
        x_big, x_crop = x
        feat_big = self.branch_big(x_big)     # [B, 2048]
        feat_crop = self.branch_crop(x_crop)  # [B, 2048]
        fused = torch.cat([feat_big, feat_crop], dim=1)  # [B, 4096]
        logits = self.fusion(fused)
        return logits


def prepare_model_fusion(category_num: int, lr: float, num_epochs: int, device, weights: torch.Tensor,
                         backbone: str = "resnet50"):
    """
    Prepare the dual-branch fusion model, optimizer, schedulers, and loss (with class weights).
    Mirrors the structure used in classify_util.prepare_model for easy drop-in.
    """
    from warmup_scheduler import GradualWarmupScheduler  # local import to mirror user's code

    model = FusionResNet(category_num, backbone=backbone)

    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)

    # Cosine schedule + warmup (same as user's original)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)

    return model, optimizer, lr_scheduler, scheduler_warmup, loss_func
