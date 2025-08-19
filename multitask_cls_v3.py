#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：multitask_cls_v3.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/8/19 上午9:08 
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader
import torchvision.models as models
from multitask_util_v3 import (TxtDataset, TwoStreamBatchSampler, update_ema, sigmoid_rampup, build_transforms,
                               evaluate_metrics, pairwise_ranking_loss, aux_soft_prior_kl,)


class MultiTaskNet(nn.Module):
    def __init__(self, backbone: str = "resnet50", feat_dim: int = 512,
                 pretrained: bool = True, dropout: float = 0.5):
        super().__init__()
        # 兼容新旧 torchvision 的预训练权重写法
        if backbone == "resnet50":
            try:
                base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                base = models.resnet50(pretrained=pretrained)
        elif backbone == "resnet18":
            try:
                base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            except Exception:
                base = models.resnet18(pretrained=pretrained)
        else:
            # 你也可以按需扩展其它骨干
            base = getattr(models, backbone)(pretrained=pretrained)

        in_feats = base.fc.in_features          # resnet18=512, resnet50=2048
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # 去掉fc，保留到avgpool
        self.feat = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_feats, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
        )
        self.head_main = nn.Linear(feat_dim, 2)  # 良/恶
        self.head_aux  = nn.Linear(feat_dim, 2)  # 非肿大/肿大

    def forward(self, x):
        x = self.backbone(x)   # [B, C, 1, 1]
        x = self.feat(x)       # [B, feat_dim]
        return self.head_main(x), self.head_aux(x)


class MeanTeacher:
    def __init__(self, model: nn.Module, ema_decay: float = 0.995):
        self.student = model
        self.teacher = type(model)()
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.ema_decay = ema_decay

    def update(self):
        update_ema(self.student, self.teacher, alpha=self.ema_decay)


def train_one_epoch(mean_teacher: MeanTeacher,
                    train_loader: DataLoader,
                    optimizer: torch.optim.Optimizer,
                    epoch: int,
                    device: torch.device,
                    rampup_epochs: int,
                    consis_weight_max: float,
                    rank_weight_max: float,
                    kl_weight_max: float,
                    ce_weight_main=None,
                    ce_weight_aux=None):

    student = mean_teacher.student
    student.train()

    ce_main = nn.CrossEntropyLoss(weight=(ce_weight_main.to(device) if ce_weight_main is not None else None),
                                  ignore_index=-1)
    ce_aux = nn.CrossEntropyLoss(weight=(ce_weight_aux.to(device) if ce_weight_aux is not None else None),
                                 ignore_index=-1)

    ramp = sigmoid_rampup(epoch, rampup_length=rampup_epochs)
    lambda_cons = consis_weight_max * ramp
    lambda_rank = rank_weight_max * ramp
    lambda_kl = kl_weight_max * ramp

    for batch in train_loader:
        # DataLoader(TxtDataset) returns: (images_weak, images_strong, labels_main, labels_aux)
        images_w, images_s, labels_main, labels_aux = batch
        images_w = images_w.to(device)
        images_s = images_s.to(device)
        labels_main = labels_main.to(device)
        labels_aux = labels_aux.to(device)

        # --- Student forward ---
        out_m_w, out_a_w = student(images_w)
        out_m_s, out_a_s = student(images_s)

        # --- Supervised (ignore label==-1) ---
        sup_main = ce_main(out_m_w, labels_main)
        sup_aux = ce_aux(out_a_w, labels_aux)

        # --- Teacher forward (no grad) ---
        with torch.no_grad():
            mean_teacher.teacher.eval()
            t_m_s, t_a_s = mean_teacher.teacher(images_s)

        # --- Main-head consistency (student strong vs teacher strong) ---
        cons_main = F.kl_div(F.log_softmax(out_m_s, dim=1),
                             F.softmax(t_m_s, dim=1), reduction='batchmean')

        losses = [sup_main, sup_aux, lambda_cons * cons_main]

        # === Cross-task regularizers for AUX-only samples (main==-1, aux!=-1) ===
        aux_only = (labels_main == -1) & (labels_aux != -1)
        if torch.any(aux_only):
            # 1) Pairwise ranking: swollen should have higher malignant logit than non-swollen
            rank_loss = pairwise_ranking_loss(out_m_w[aux_only], labels_aux[aux_only])
            if rank_loss is not None:
                losses.append(lambda_rank * rank_loss)

            # 2) Soft-prior KL: pull main prob toward prior implied by AUX label/teacher AUX
            with torch.no_grad():
                p_aux_teacher = F.softmax(t_a_s[aux_only], dim=1)[:, 1]  # p(swollen)
            kl_loss = aux_soft_prior_kl(out_m_w[aux_only], labels_aux[aux_only], p_aux_teacher,
                                        q_mal_when_swollen=0.65, q_mal_when_non=0.35, mix=0.7, T=1.0)
            losses.append(lambda_kl * kl_loss)

        loss = torch.stack([l for l in losses]).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        mean_teacher.update()


def validate(model: nn.Module, valid_loader: DataLoader, device: torch.device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in valid_loader:
            images_w, _, labels_main, _ = batch
            images_w = images_w.to(device)
            labels_main = labels_main.to(device)
            out_m, _ = model(images_w)
            pred = out_m.argmax(dim=1).cpu().numpy()
            mask = (labels_main != -1).cpu().numpy()
            ys.extend(labels_main.cpu().numpy()[mask])
            ps.extend(pred[mask])
    cm, report, macro_f1, bal_acc = evaluate_metrics(ys, ps)
    return macro_f1, bal_acc, report, cm


def main():
    parser = argparse.ArgumentParser()
    # Provide your txt files: lines are "path label_main label_aux"; for aux-only, set label_main=-1
    parser.add_argument('--train_labeled_txt', type=str, required=True,
                        help='txt with path label_main label_aux for supervised set')
    parser.add_argument('--train_auxonly_txt', type=str, required=True,
                        help='txt with path -1 label_aux for aux-only set')
    parser.add_argument('--val_txt', type=str, required=True,
                        help='txt with path label_main label_aux for validation')
    parser.add_argument('--test_txt', type=str, default=None,
                        help='optional test txt with path label_main label_aux')
    parser.add_argument('--device_id', type=str, help="gpu device id to use", default='2')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--ema', type=float, default=0.995)
    parser.add_argument('--rampup', type=int, default=50)
    parser.add_argument('--consw', type=float, default=1.0)
    parser.add_argument('--rankw', type=float, default=0.5)
    parser.add_argument('--klw', type=float, default=0.3)
    parser.add_argument('--labeled_ratio', type=float, default=0.5,
                        help='portion of labeled samples per batch')
    parser.add_argument('--steps_per_epoch', type=int, default=1000)
    parser.add_argument('--save', type=str, default='best.pt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    weak_tf, strong_tf = build_transforms(args.img_size)

    # Datasets from TXT (your original format)
    ds_train_l = TxtDataset(args.train_labeled_txt, transform_w=weak_tf, transform_s=strong_tf)
    ds_train_w = TxtDataset(args.train_auxonly_txt, transform_w=weak_tf, transform_s=strong_tf)
    ds_val = TxtDataset(args.val_txt, transform_w=weak_tf, transform_s=strong_tf)
    ds_test = TxtDataset(args.test_txt, transform_w=weak_tf, transform_s=strong_tf) if args.test_txt else None

    train_ds = ConcatDataset([ds_train_l, ds_train_w])

    # Build two-stream batch sampler: indices are into the ConcatDataset
    labeled_indices = list(range(0, len(ds_train_l)))
    aux_only_indices = list(range(len(ds_train_l), len(ds_train_l) + len(ds_train_w)))
    batch_sampler = TwoStreamBatchSampler(labeled_indices, aux_only_indices,
                                          batch_size=args.bs,
                                          labeled_ratio=args.labeled_ratio,
                                          steps_per_epoch=args.steps_per_epoch)

    train_loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(ds_val, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = (DataLoader(ds_test, batch_size=args.bs, shuffle=False, num_workers=4, pin_memory=True)
                   if ds_test is not None else None)

    model = MultiTaskNet()
    mean_teacher = MeanTeacher(model, ema_decay=args.ema)
    mean_teacher.student.to(device); mean_teacher.teacher.to(device)

    # Choose ONE imbalance strategy: weighted CE (do not oversample again here)
    class_weights_main = torch.tensor([1.0, 1.0])  # adjust if needed
    class_weights_aux = torch.tensor([1.0, 1.0])

    optimizer = torch.optim.AdamW(mean_teacher.student.parameters(), lr=args.lr, weight_decay=1e-4)

    best_bal = -1.0
    for epoch in range(1, args.epochs + 1):
        train_one_epoch(mean_teacher, train_loader, optimizer, epoch, device,
                        rampup_epochs=args.rampup,
                        consis_weight_max=args.consw,
                        rank_weight_max=args.rankw,
                        kl_weight_max=args.klw,
                        ce_weight_main=class_weights_main,
                        ce_weight_aux=class_weights_aux)

        macro_f1, bal_acc, report, cm = validate(mean_teacher.teacher, val_loader, device)
        print(f"Epoch {epoch}: macroF1={macro_f1:.4f}  balancedAcc={bal_acc:.4f}")
        print("Confusion Matrix:", cm)
        print("Classification Report:", report)

        if bal_acc > best_bal:
            best_bal = bal_acc
            torch.save({'epoch': epoch, 'model': mean_teacher.teacher.state_dict(),
                        'best_balanced_acc': best_bal}, args.save)
            print(f"[Saved] best model @ epoch {epoch}, balanced acc={best_bal:.4f}")

    if test_loader is not None:
        macro_f1, bal_acc, report, cm = validate(mean_teacher.teacher, test_loader, device)
        print("==== TEST ====")
        print(f"macroF1={macro_f1:.4f}  balancedAcc={bal_acc:.4f}")
        print("Confusion Matrix:", cm)
        print("Classification Report:", report)


if __name__ == '__main__':
    main()
