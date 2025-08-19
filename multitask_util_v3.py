#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：multitask_util_v3.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/8/19 上午9:09 
"""
import os
import math
import random
from typing import List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, Sampler
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, balanced_accuracy_score


def update_ema(student: nn.Module, teacher: nn.Module, alpha: float = 0.995):
    with torch.no_grad():
        for p_t, p_s in zip(teacher.parameters(), student.parameters()):
            p_t.data.mul_(alpha).add_(p_s.data, alpha=1.0 - alpha)


def sigmoid_rampup(current: int, rampup_length: int):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(math.exp(-5.0 * phase * phase))


class TxtDataset(Dataset):
    """txt 每行: path label_main label_aux (若无主标签则 label_main=-1)"""
    def __init__(self, txt_file, transform_w=None, transform_s=None):
        self.samples = []
        with open(txt_file, 'r') as f:
            for line in f:
                path, m, a = line.strip().split(',')
                self.samples.append({"path": os.path.join('/mnt/disk1/caoxu/dataset/中山淋巴结/训练集', path), "label_main": int(m), "label_aux": int(a)})
        self.tw = transform_w
        self.ts = transform_s if transform_s is not None else transform_w

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        it = self.samples[int(idx)]
        img = Image.open(it["path"]).convert("RGB")
        img_w = self.tw(img) if self.tw else img
        img_s = self.ts(img) if self.ts else img
        return (
            img_w, img_s,
            torch.tensor(it["label_main"], dtype=torch.long),
            torch.tensor(it["label_aux"], dtype=torch.long),
        )


class TwoStreamBatchSampler(Sampler[List[int]]):
    def __init__(self, labeled_indices: List[int], aux_only_indices: List[int],
                 batch_size: int, labeled_ratio: float = 0.5, steps_per_epoch: int = 1000):
        self.labeled = np.array(labeled_indices)
        self.aux_only = np.array(aux_only_indices)
        self.bs = batch_size
        self.k_l = max(1, int(round(batch_size * labeled_ratio)))
        self.k_a = max(0, batch_size - self.k_l)
        self.steps = steps_per_epoch

    def __len__(self):
        return self.steps

    def __iter__(self):
        for _ in range(self.steps):
            idx_l = np.random.choice(self.labeled, self.k_l, replace=True)
            idx_a = np.random.choice(self.aux_only, self.k_a, replace=True) if self.k_a > 0 else []
            batch = np.concatenate([idx_l, idx_a]).tolist()
            random.shuffle(batch)
            yield batch


def build_transforms(img_size: int = 224):
    import torchvision.transforms as T
    weak = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    strong = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return weak, strong


def evaluate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    report = classification_report(y_true, y_pred, labels=[0,1], target_names=["良性","恶性"], digits=4)
    macro_f1 = f1_score(y_true, y_pred, labels=[0,1], average='macro')
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return cm, report, macro_f1, bal_acc


def pairwise_ranking_loss(main_logits: torch.Tensor, aux_labels: torch.Tensor, margin: float = 0.2):
    malignant_score = main_logits[:, 1]
    swollen_idx = (aux_labels == 1).nonzero(as_tuple=True)[0]
    non_swollen_idx = (aux_labels == 0).nonzero(as_tuple=True)[0]
    if len(swollen_idx) == 0 or len(non_swollen_idx) == 0:
        return None
    k = min(64, len(swollen_idx))
    i = swollen_idx[torch.randint(len(swollen_idx), (k,))]
    j = non_swollen_idx[torch.randint(len(non_swollen_idx), (k,))]
    diff = malignant_score[i] - malignant_score[j]
    return torch.relu(margin - diff).mean()


def aux_soft_prior_kl(main_logits: torch.Tensor,
                      aux_hard: torch.Tensor,
                      aux_soft: Optional[torch.Tensor] = None,
                      q_mal_when_swollen: float = 0.65,
                      q_mal_when_non: float = 0.35,
                      mix: float = 0.7,
                      T: float = 1.0):
    with torch.no_grad():
        if aux_soft is None:
            p_swollen = aux_hard.float()
        else:
            p_swollen = mix * aux_hard.float() + (1 - mix) * aux_soft.detach()
        q_mal = p_swollen * q_mal_when_swollen + (1 - p_swollen) * q_mal_when_non
        q = torch.stack([1.0 - q_mal, q_mal], dim=1).clamp(1e-6, 1 - 1e-6)
    log_p = torch.log_softmax(main_logits / T, dim=1)
    return nn.functional.kl_div(log_p, q, reduction='batchmean')
