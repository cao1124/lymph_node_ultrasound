#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：multitask_one_cls_aux.py
@IDE     ：PyCharm
@Author  ：cao xu (edited)
@Date    ：2025/9/29

说明（关键改动）：
- FIX: 教师用弱增强、学生用强增强做主任务一致性（基于CE伪标签 + 置信度阈值）
- FIX: 仅调用 warmup.step()，不再双步 LR
- FIX: 训练采用“双 DataLoader”（labeled 与 weak 分开），每 step 固定比例融合
- 调整数据增强已在 util 中完成
"""
from multitask_one_cls_aux_util import MetastasisCls2CN

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：multitask_cls.py (One-Class Aux)
@IDE     ：PyCharm
@Author  ：cao xu (edited for One-Class Aux)
@Date    ：2025/9/29

说明（关键改动）：
- 主任务：恶性结节的【转移 vs 未转移】分类（小图）。
- 辅任务：仅“肿大”大图，一类学习（Deep-SVDD 风格），不需要“非肿大”。
- 总损失： L_main_CE + cw_main*L_main_cons + cw_aux*L_svdd + λ_var*L_var (+ 可选 L_aux_view_consistency)。
- 仍采用双 DataLoader（主：有 main_label；辅：仅正样本），Mean Teacher 只用于主任务一致性。
"""
import warnings
from itertools import cycle
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, SubsetRandomSampler
from multitask_one_cls_aux_util import *  # noqa
matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    current = np.clip(current, 0.0, rampup_length)
    phase = 1.0 - current / rampup_length
    return float(np.exp(-5.0 * phase * phase))  # [0,1]


def semi_supervised_train(
    data_dir, labeled_txt_path, weak_label_txt_path,
    num_epochs, bs, pt_dir, category_num_main,
    model_name, device, lr, target_list_main, patience,
    # Mean Teacher 与主任务一致性超参：
    alpha=0.995, consistency_weight=0.1, tau=0.95, rampup_epochs=60,
    labeled_ratio=0.7,
    # 一类辅助任务超参：
    aux_embed_dim=128, lambda_aux=0.5, lambda_var=0.1, aux_center_momentum=0.99,
    aux_sigma_min=0.1, aux_rampup_epochs=40, lambda_aux_cons=0.0
):
    """
    - weak_label_txt_path: 仅包含“肿大”样本（大图）。可无标签列。
    - target_list_main: ["未转移","转移"]
    - lambda_aux_cons: 辅任务 weak/strong 视图一致性权重（0 则关闭）。
    """
    # 1) 读取数据列表
    with open(labeled_txt_path, 'r', encoding='utf-8') as f:
        labeled_lines = f.readlines()
    with open(weak_label_txt_path, 'r', encoding='utf-8') as f:
        weak_label_lines = f.readlines()

    # 2) 五折（固定第5折为测试）
    labeled_lines = shuffle(labeled_lines, random_state=1)
    fold_size = max(1, int(len(labeled_lines) * 0.2))
    folds = [labeled_lines[i:i + fold_size] for i in range(0, len(labeled_lines), fold_size)]
    if len(folds) < 5:
        # 不足5折时，最后一折作为测试，前一折验证，其他训练
        while len(folds) < 5:
            folds.append([])
    test_list = [folds[4]]

    for fold_idx in range(4):
        print(f'五折交叉验证 第{fold_idx + 1}次实验')

        train_list, valid_list = [], []
        for idx, fold in enumerate(folds):
            if idx == fold_idx:
                valid_list.append(fold)
            elif fold != test_list[0]:
                train_list.append(fold)

        # ---- 构建数据集 ----
        train_labeled_dataset = WeaklyLabeledDataset(
            [item for fold in train_list for item in fold],
            data_dir,
            transform=img_trans['train'],
            strong_transform=img_trans['strong'],
            has_main_label=True,
            has_aux_label=False
        )
        weak_label_dataset = WeaklyLabeledDataset(
            weak_label_lines,
            data_dir,
            transform=img_trans['train'],
            strong_transform=img_trans['strong'],
            has_main_label=False,
            has_aux_label=True  # 仅肿大正样本；标签值不会用于CE
        )
        valid_dataset = WeaklyLabeledDataset(
            [item for fold in valid_list for item in fold],
            data_dir,
            transform=img_trans['valid'],
            has_main_label=True,
            has_aux_label=False
        )
        test_dataset = WeaklyLabeledDataset(
            [item for fold in test_list for item in fold],
            data_dir,
            transform=img_trans['valid'],
            has_main_label=True,
            has_aux_label=False
        )

        print(f'train(labeled)={len(train_labeled_dataset)}, train(weak=swollen+)={len(weak_label_dataset)}, '
              f'valid={len(valid_dataset)}, test={len(test_dataset)}')

        # ---- DataLoader：分流并控制比例 ----
        # 直接用数据集里的标签，避免字符串解析不一致
        labeled_labels = train_labeled_dataset.labels_main if train_labeled_dataset.labels_main is not None else []
        class_weights = None  # 可按需开启类别权重

        # 平衡采样（仅 labeled；忽略 -1）
        if labeled_labels and any([t in (0, 1) for t in labeled_labels]):
            sampler_lbl = SubsetRandomSampler(BalanceDataSampler(labeled_labels).prob)
        else:
            sampler_lbl = None

        # 每 step 采样比例
        bs_labeled = max(1, int(bs * labeled_ratio))
        bs_weak = max(1, bs - bs_labeled)

        labeled_loader = DataLoader(
            train_labeled_dataset,
            batch_size=bs_labeled,
            shuffle=(sampler_lbl is None),
            sampler=sampler_lbl,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        weak_loader = DataLoader(
            weak_label_dataset,
            batch_size=bs_weak,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        valid_loader = DataLoader(valid_dataset, batch_size=bs, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=2)

        # 3) 模型与优化器
        model, optimizer, cosine, warmup, loss_func_main = prepare_multi_task_model(
            category_num_main=category_num_main,
            model_name=model_name,
            lr=lr,
            num_epochs=num_epochs,
            device=device,
            aux_embed_dim=aux_embed_dim,
            weights_main=None
        )

        mean_teacher = MeanTeacherModel(model).to(device)
        early_stopping = EarlyStopping(pt_dir, patience=patience)

        best_valid_acc, best_epoch = 0.0, 0
        history = {'train_loss': [], 'valid_loss': []}

        # 一类中心 c（在训练过程中 EMA 更新）
        aux_center = None

        # 4) 训练循环
        for epoch in range(num_epochs):
            model.train()
            print(f"Epoch: {epoch + 1}/{num_epochs}")

            # ramp-up 权重
            cw_main = consistency_weight * sigmoid_rampup(epoch, rampup_epochs)
            cw_aux = lambda_aux * sigmoid_rampup(epoch, aux_rampup_epochs)

            it_l = iter(cycle(labeled_loader))
            it_w = iter(cycle(weak_loader))
            # 为覆盖更大数据端，按两者长度较大的 loader 的步数来迭代
            steps = max(len(labeled_loader), len(weak_loader))

            running_loss = 0.0
            main_correct = 0
            total_main = 0

            for _ in range(steps):
                batch_l = next(it_l)
                batch_w = next(it_w)

                # ---- 有主标签批（监督 CE）----
                images_l = batch_l['image'].to(device)         # 弱增强
                labels_main = batch_l['label_main'].to(device) # 0/1（未转移/转移）
                outputs_main_l, _ = model(images_l)
                loss_main = loss_func_main(outputs_main_l, labels_main)

                with torch.no_grad():
                    preds_main = outputs_main_l.argmax(dim=1)
                    main_correct += (preds_main == labels_main).sum().item()
                    total_main += labels_main.size(0)

                # ---- 仅辅任务批（一类学习 + 可选视图一致性）----
                images_w_weak = batch_w['image'].to(device)           # 教师看弱增强（用于主任务一致性伪标签）
                images_w_strong = batch_w['image_strong'].to(device)  # 学生看强增强

                # 学生前向：拿到 z_aux（用于一类学习）
                _, z_aux_wk = model(images_w_weak)     # weak 视图
                _, z_aux_st = model(images_w_strong)   # strong 视图（仅用于一致性）

                # 归一化以稳定几何尺度
                z_aux_wk = F.normalize(z_aux_wk, dim=1)
                z_aux_st = F.normalize(z_aux_st, dim=1)

                # 初始化 / EMA 更新一类中心
                with torch.no_grad():
                    batch_center = z_aux_wk.detach().mean(dim=0)
                    if aux_center is None:
                        aux_center = batch_center
                    else:
                        aux_center = aux_center_momentum * aux_center + (1 - aux_center_momentum) * batch_center

                # Deep-SVDD 一类损失：把正样本拉向中心
                loss_svdd = ((z_aux_wk - aux_center) ** 2).sum(dim=1).mean()

                # 防塌缩：每维标准差至少为 aux_sigma_min
                std_per_dim = z_aux_wk.std(dim=0)
                loss_var = torch.clamp_min(aux_sigma_min - std_per_dim, 0).mean()

                # 可选：辅任务 weak/strong 视图一致性
                loss_aux_cons = F.mse_loss(z_aux_wk, z_aux_st) if lambda_aux_cons > 0 else 0.0

                # ---- 主任务一致性（教师弱增强 -> 伪标签；学生强增强 -> 对齐）----
                with torch.no_grad():
                    teacher_main_w, _ = mean_teacher(images_w_weak, is_teacher=True)
                    probs_t = F.softmax(teacher_main_w, dim=1)
                    conf, pseudo = probs_t.max(dim=1)
                    mask = conf.ge(tau)

                # 简单类均衡截断（每类最多取 K 个）
                K = 16  # 可按需调整
                if mask.any():
                    idx_all = torch.nonzero(mask).squeeze(1)
                    keep_idx = []
                    for c in [0, 1]:
                        cls_idx = idx_all[pseudo[idx_all] == c]
                        if cls_idx.numel() > 0:
                            keep_idx.append(cls_idx[:K])
                    if len(keep_idx) > 0:
                        keep_idx = torch.cat(keep_idx)
                        student_main_s, _ = model(images_w_strong[keep_idx])
                        loss_cons = F.cross_entropy(student_main_s, pseudo[keep_idx])
                    else:
                        loss_cons = torch.tensor(0.0, device=device)
                else:
                    loss_cons = torch.tensor(0.0, device=device)

                # 总损失：主任务监督 + 主一致性 + 辅任务一类 + 方差正则 (+ 辅视图一致性)
                loss = (
                    loss_main
                    + cw_main * loss_cons
                    + cw_aux  * loss_svdd
                    + lambda_var * loss_var
                    + (lambda_aux_cons * loss_aux_cons if lambda_aux_cons > 0 else 0.0)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                mean_teacher.update_teacher(alpha=alpha)

                running_loss += float(loss.item())

            # 只调用 warmup（内部会在合适 epoch 驱动 cosine）
            warmup.step()

            avg_train_loss = running_loss / steps if steps > 0 else 0.0
            main_acc = main_correct / total_main if total_main > 0 else 0.0
            print(f"Train: Loss {avg_train_loss:.4f} | MainAcc {main_acc:.4f} | CW_main {cw_main:.3f} | CW_aux {cw_aux:.3f}")

            # 验证（用 Teacher）
            valid_loss, valid_main_acc = validate(
                mean_teacher.teacher, valid_loader, device, loss_func_main
            )
            print(f"Val:   Loss {valid_loss:.4f} | Acc {valid_main_acc:.4f}")

            history['train_loss'].append(avg_train_loss)
            history['valid_loss'].append(valid_loss)

            if valid_main_acc > best_valid_acc:
                best_valid_acc = valid_main_acc
                best_epoch = epoch + 1
                os.makedirs(pt_dir, exist_ok=True)
                torch.save(mean_teacher.teacher.state_dict(),
                           os.path.join(pt_dir, f'fold{fold_idx}-best-acc-model.pt'))
                print(f"New best model saved @ epoch {best_epoch} acc={best_valid_acc:.4f}")

            early_stopping(valid_loss, mean_teacher.teacher)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 5) 保存曲线
        os.makedirs(pt_dir, exist_ok=True)
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Valid Loss')
        plt.legend()
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Training History')
        plt.savefig(os.path.join(pt_dir, f'loss_curve_fold{fold_idx}.png'))
        plt.close()

        # 6) 测试（加载最佳 Teacher 权重到同构模型）
        test_model = prepare_multi_task_model(
            category_num_main, model_name, lr, num_epochs, device, aux_embed_dim
        )[0]
        test_model.load_state_dict(torch.load(os.path.join(pt_dir, f'fold{fold_idx}-best-acc-model.pt')))
        test_model.to(device)

        test_acc, test_report = evaluate(test_model, test_loader, device, target_list_main)
        print(f"Test Accuracy: {test_acc:.4f}")
        print("Classification Report:")
        print(test_report)

        with open(os.path.join(pt_dir, f'test_report_fold{fold_idx}.txt'), 'w', encoding='utf-8') as f:
            f.write(test_report)


def validate(model, loader, device, loss_func):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels_main = batch['label_main'].to(device)
            outputs_main, _ = model(images)
            loss = loss_func(outputs_main, labels_main)
            total_loss += float(loss.item())
            preds = outputs_main.argmax(dim=1)
            correct += (preds == labels_main).sum().item()
            total += labels_main.size(0)
    return total_loss / len(loader), (correct / total if total > 0 else 0.0)


def evaluate(model, loader, device, target_names):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels_main = batch['label_main'].to(device)
            outputs_main, _ = model(images)
            preds = outputs_main.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_main.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print("Confusion Matrix:")
    print(cm)
    return accuracy, report


def classification():
    # === 路径与数据列表 ===
    # 请替换为你的小图（主任务：转移/未转移）与大图（辅任务：仅肿大）列表路径
    data_dir = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集/'
    labeled_txt_path = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/crop/20250625-中山淋巴恶性瘤淋巴瘤2分类-补充训练-crop.txt'
    weak_label_txt_path = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/ori/20250929-仅肿大.txt'

    # 主任务标签名（用于分类报告）
    target_list_main = [x.name for x in MetastasisCls2CN]  # ['未转移', '转移']

    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'
    category_num_main = len(target_list_main)  # 2
    bs = 64
    lr = 0.001
    patience = 20
    num_epochs = 500

    run_tag = '20250929-半监督-转移分类-辅任务OneClass-'
    pt_dir = os.path.join('Section_QC', run_tag + f'{model_name}-bs{bs}-lr{lr}')
    os.makedirs(pt_dir, exist_ok=True)

    print(f'开始训练: 主任务=转移/未转移({category_num_main}类), 辅任务=One-Class(仅肿大)')
    print(f'模型: {model_name}, batch size={bs}, 学习率={lr}')

    semi_supervised_train(
        data_dir=data_dir,
        labeled_txt_path=labeled_txt_path,
        weak_label_txt_path=weak_label_txt_path,
        num_epochs=num_epochs,
        bs=bs,
        pt_dir=pt_dir,
        category_num_main=category_num_main,
        model_name=model_name,
        device=device,
        lr=lr,
        target_list_main=target_list_main,
        patience=patience,
        # Mean Teacher / 主一致性
        alpha=0.995,
        consistency_weight=0.1,
        tau=0.95,
        rampup_epochs=60,
        labeled_ratio=0.7,
        # One-Class 辅助任务
        aux_embed_dim=128,
        lambda_aux=0.5,
        lambda_var=0.1,
        aux_center_momentum=0.99,
        aux_sigma_min=0.1,
        aux_rampup_epochs=40,
        lambda_aux_cons=0.0  # 如需启用视图一致性，请设为 >0
    )
    print('训练完成')


if __name__ == '__main__':
    classification()
