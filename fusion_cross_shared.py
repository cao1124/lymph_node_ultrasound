#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：fusion_cross_shared.py.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/11/12 下午3:50

Cross-fusion model for dual images (big + crop) following the new diagram:
Big/Crop -- SharedEncoder --> {BaseEncoder, DetailEncoder} (shared per-type)
    Cross Fusions: F1 = fuse(Big-Detail, Crop-Base), F2 = fuse(Big-Base, Crop-Detail)
    Head: concat(F1, F2) -> classifier

Drop-in replacement for the previous FusionResNet:
  - forward expects a tuple (big_batch, crop_batch)
  - optimizer/scheduler setup same as before
"""

from typing import Tuple, Union, List
import torch
import torch.nn as nn
from torchvision import models

Tensor = torch.Tensor
PairBatch = Union[Tuple[Tensor, Tensor], List[Tensor]]


# ---------- Small building blocks ----------
class GAP(nn.Module):
    def __init__(self): super().__init__(); self.pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x): return self.pool(x).flatten(1)


class SEModule(nn.Module):
    """Squeeze-Excitation for detail enhancement."""
    def __init__(self, ch: int, r: int = 16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // r, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(ch // r, ch, 1, bias=False), nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w


class DetailBlock(nn.Module):
    """
    Light detail extractor: dilated convs + SE.
    Input:  C=256 (ResNet layer1 output), keep relatively high resolution.
    Output: C=512 feature map.
    """
    def __init__(self, in_ch: int = 256, mid: int = 256, out_ch: int = 512):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=2, dilation=2, bias=False), nn.BatchNorm2d(mid), nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            SEModule(out_ch)
        )

    def forward(self, x): return self.body(x)


class FusionUnit(nn.Module):
    """
    Fuse two pooled vectors by concatenation -> BN+Dropout+FC -> ReLU+BN.
    in_dim = a_dim + b_dim
    """
    def __init__(self, in_dim: int, hid: int = 512, p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(p),
            nn.Linear(in_dim, hid),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(hid)
        )

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        return self.net(torch.cat([a, b], dim=1))


# ---------- Main Model ----------
class CrossFusionResNetShared(nn.Module):
    """
    ResNet50-based cross fusion with shared stem:
      SharedEncoder: conv1-bn1-relu-maxpool-layer1
      BaseEncoder  : layer2-layer3
      DetailEncoder: DetailBlock (+ shallow conv)  -> project to 512
      Fusions      : F1 = fuse(Big-Detail, Crop-Base), F2 = fuse(Big-Base, Crop-Detail)
      Head         : concat(F1, F2) -> FC -> logits
    """
    def __init__(self, num_classes: int = 6, pretrained: bool = True):
        super().__init__()

        # ---- Build a reference ResNet50 and split to stages
        res = models.resnet50(pretrained=pretrained)

        # Shared encoder (stem + layer1): output C=256
        self.shared = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool, res.layer1
        )

        # Base encoder (semantic): layer2 + layer3 (C: 512 -> 1024)
        self.base_encoder = nn.Sequential(res.layer2, res.layer3)

        # For detail path we DON'T reuse layer2/layer3 to keep specialization.
        # Use a light detail block then a shallow residual stack to 512.
        self.detail_encoder = DetailBlock(in_ch=256, mid=256, out_ch=512)

        # A small conv stack to align base path to 1024 feat map before GAP
        self.base_align = nn.Identity()   # layer3 already outputs 1024 ch

        # Global average pooling + projection to 512-d
        self.gap = GAP()
        self.proj_base   = nn.Linear(1024, 512)
        self.proj_detail = nn.Linear(512, 512)

        # Cross fusion units
        # Each unit sees concat(512, 512) -> 1024 -> 512
        self.fuse_F1 = FusionUnit(in_dim=1024, hid=512)  # Big-Detail ⊕ Crop-Base
        self.fuse_F2 = FusionUnit(in_dim=1024, hid=512)  # Big-Base   ⊕ Crop-Detail

        # Head: concat(F1, F2) -> 1024 -> 512 -> num_classes
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 2, 512), nn.ReLU(inplace=True), nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    # ----- per-branch forward -----
    def _encode(self, x: Tensor):
        """Run one image through shared -> (base_feat, detail_feat) [both projected to 512-d]."""
        s = self.shared(x)                      # [B,256,H/4,W/4]
        b_map = self.base_encoder(s)            # [B,1024,H/16,W/16]
        d_map = self.detail_encoder(s)          # [B,512, H/4, W/4]

        b_vec = self.proj_base(self.gap(b_map))     # [B,512]
        d_vec = self.proj_detail(self.gap(d_map))   # [B,512]
        return b_vec, d_vec

    def forward(self, inputs: PairBatch) -> Tensor:
        assert isinstance(inputs, (tuple, list)) and len(inputs) == 2, \
            "Expect tuple (big_batch, crop_batch)"
        big, crop = inputs

        # Encode two images with shared weights (shared stem, shared type-specific encoders)
        big_base,  big_detail  = self._encode(big)
        crop_base, crop_detail = self._encode(crop)

        # Cross fusions
        F1 = self.fuse_F1(big_detail, crop_base)   # 图中上路的 F
        F2 = self.fuse_F2(big_base,  crop_detail)  # 图中下路的 F

        # Head
        fused = torch.cat([F1, F2], dim=1)         # [B, 1024]
        logits = self.head(fused)
        return logits


# ---------- Prepare function ----------
def prepare_model_fusion_cross(num_classes: int, lr: float, num_epochs: int, device,
                               class_weights: Tensor, pretrained: bool = True):
    """
    Return: model, optimizer, cosine_scheduler, warmup_scheduler, loss_func
    与旧版接口保持一致，便于直接替换。
    """
    from warmup_scheduler import GradualWarmupScheduler
    import torch.optim as optim

    model = CrossFusionResNetShared(num_classes, pretrained=pretrained)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.to(device)

    loss_func = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=2e-4, momentum=0.9, nesterov=True)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.005)
    warmup  = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=cosine)

    return model, optimizer, cosine, warmup, loss_func
