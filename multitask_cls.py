#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：multitask_cls.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/8/15 下午2:21 
"""
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report
from multitask_util import *
from torch.utils.data import DataLoader, ConcatDataset, SubsetRandomSampler
import matplotlib
import matplotlib.pyplot as plt
import torch.multiprocessing
import warnings
import numpy as np
import torch.nn.functional as F
import os
import torch

matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


# ================== 半监督训练函数 ==================
def semi_supervised_train(data_dir, labeled_txt_path, weak_label_txt_path,
                          num_epochs, bs, pt_dir, category_num_main, category_num_aux,
                          model_name, device, lr, target_list_main, target_list_aux,
                          patience, alpha=0.99, consistency_weight=5.0):
    # 1. 数据准备
    # 加载有标签数据（良恶性分类数据）
    with open(labeled_txt_path, 'r', encoding='utf-8') as file:
        labeled_lines = file.readlines()

    # 加载弱标签数据（肿大状态分类数据）
    with open(weak_label_txt_path, 'r', encoding='utf-8') as file:
        weak_label_lines = file.readlines()

    # 五折交叉验证 (仅使用有标签数据)
    labeled_lines = shuffle(labeled_lines, random_state=1)
    fold_size = int(len(labeled_lines) * 0.2)
    folds = [labeled_lines[i:i + fold_size] for i in range(0, len(labeled_lines), fold_size)]

    # 固定测试集为第五折
    test_list = [folds[4]]

    for fold_idx in range(4):
        print(f'五折交叉验证 第{fold_idx + 1}次实验')

        # 构建训练/验证/测试集
        train_list, valid_list = [], []
        for idx, fold in enumerate(folds):
            if idx == fold_idx:
                valid_list.append(fold)
            elif fold != test_list[0]:
                train_list.append(fold)

        # ==== 重构数据集创建 ====
        # 有标签训练集（只有主任务标签 - 良恶性）
        train_labeled_dataset = WeaklyLabeledDataset(
            [item for fold in train_list for item in fold],
            data_dir,
            transform=img_trans['train'],
            strong_transform=img_trans['strong'],
            has_main_label=True,
            has_aux_label=False  # 病理数据没有肿大标签
        )

        # 弱标签数据集（只有辅任务标签 - 肿大状态）
        weak_label_dataset = WeaklyLabeledDataset(
            weak_label_lines,
            data_dir,
            transform=img_trans['train'],
            strong_transform=img_trans['strong'],
            has_main_label=False,  # 无良恶性标签
            has_aux_label=True  # 有肿大标签
        )

        # 合并数据集
        train_dataset = ConcatDataset([train_labeled_dataset, weak_label_dataset])

        # 验证集（只有主任务标签）
        valid_dataset = WeaklyLabeledDataset(
            [item for fold in valid_list for item in fold],
            data_dir,
            transform=img_trans['valid'],
            has_main_label=True,
            has_aux_label=False
        )

        # 测试集（只有主任务标签）
        test_dataset = WeaklyLabeledDataset(
            [item for fold in test_list for item in fold],
            data_dir,
            transform=img_trans['valid'],
            has_main_label=True,
            has_aux_label=False
        )

        # 数据集大小
        train_size = len(train_dataset)
        valid_size = len(valid_dataset)
        test_size = len(test_dataset)
        print(f'train_size: {train_size}, valid_size: {valid_size}, test_size: {test_size}')

        # 2. 数据加载器
        # 有标签数据的类权重（仅主任务）
        labeled_labels = [1 if line.strip().split(',')[1] == '恶性' else 0 for line in [item for fold in train_list for item in fold]]
        label_count = Counter(labeled_labels)
        class_weights = [1.0 / label_count.get(i, 1.0) for i in range(category_num_main)]

        # 采样器（仅对有标签数据）
        # sampler = BalanceDataSampler(labeled_labels) if len(labeled_labels) > 0 else None
        if len(labeled_labels) > 0:
            base_sampler = BalanceDataSampler(labeled_labels)  # 返回的索引针对 labeled 子数据集 (0..L-1)
            # 将这些索引映射到 ConcatDataset（[labeled, weak]）的全局索引空间
            labeled_count = len(train_labeled_dataset)
            weak_count = len(weak_label_dataset)
            # base_sampler.prob 是一个列表，包含重复的 labeled 索引
            combined_indices = list(base_sampler.prob)  # indices referencing 0..labeled_count-1
            # 把弱标签数据的索引也加入（offset by labeled_count），保证弱样本出现在采样池中
            combined_indices += list(range(labeled_count, labeled_count + weak_count))
            # 随机打乱顺序，形成最终采样序列
            random.shuffle(combined_indices)
            # 使用 SubsetRandomSampler 把我们准备好的索引列表传入 DataLoader
            sampler = SubsetRandomSampler(combined_indices)
        else:
            sampler = None

        train_loader = DataLoader(
            train_dataset,
            batch_size=bs,
            shuffle=sampler is None,
            num_workers=4,
            pin_memory=True,
            sampler=sampler
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=2
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=bs,
            shuffle=False,
            num_workers=2
        )

        # 3. 模型准备
        model, optimizer, scheduler, warmup, loss_func_main, loss_func_aux = prepare_multi_task_model(
            category_num_main=category_num_main,
            category_num_aux=category_num_aux,
            model_name=model_name,
            lr=lr,
            num_epochs=num_epochs,
            device=device,
            weights_main=torch.tensor(class_weights) if class_weights is not None else None)  # 确保 weights_main 不是 None

        # 创建Mean Teacher模型
        mean_teacher = MeanTeacherModel(model)
        mean_teacher.to(device)

        # 早停机制
        early_stopping = EarlyStopping(pt_dir, patience=patience)

        # 训练变量
        best_valid_acc = 0.0
        best_epoch = 0
        history = {'train_loss': [], 'valid_loss': []}

        # 4. 训练循环
        for epoch in range(num_epochs):
            print(f"Epoch: {epoch + 1}/{num_epochs}")
            model.train()

            # 训练统计
            train_loss = 0.0
            main_correct = 0
            aux_correct = 0
            total_labeled_main = 0
            total_labeled_aux = 0

            for batch in train_loader:
                images = batch['image'].to(device)
                images_strong = batch['image_strong'].to(device)
                labels_main = batch['label_main'].to(device)
                labels_aux = batch['label_aux'].to(device)

                # 区分样本类型
                labeled_main_mask = (labels_main != -1)  # 有主标签的样本（病理数据）
                labeled_aux_mask = (labels_aux != -1)  # 有辅标签的样本（肿大数据）
                unlabeled_mask = ~labeled_main_mask  # 无主标签的样本（肿大数据）

                # 学生模型预测 (弱增强)
                outputs_main, outputs_aux = model(images)

                # 教师模型预测 (无梯度)
                with torch.no_grad():
                    mean_teacher.teacher.eval()
                    teacher_main, teacher_aux = mean_teacher(images_strong, is_teacher=True)

                # 损失计算
                losses = []

                # === 主任务监督损失（病理数据）===
                if torch.any(labeled_main_mask):
                    loss_main = loss_func_main(
                        outputs_main[labeled_main_mask],
                        labels_main[labeled_main_mask]
                    )
                    losses.append(loss_main)

                    # 统计准确率
                    _, main_preds = torch.max(outputs_main[labeled_main_mask], 1)
                    main_correct += torch.sum(main_preds == labels_main[labeled_main_mask]).item()
                    total_labeled_main += torch.sum(labeled_main_mask).item()

                # === 辅任务监督损失（肿大数据）===
                if torch.any(labeled_aux_mask):
                    loss_aux = loss_func_aux(
                        outputs_aux[labeled_aux_mask],
                        labels_aux[labeled_aux_mask]
                    )
                    losses.append(0.5 * loss_aux)  # 辅任务权重

                    # 统计准确率
                    with torch.no_grad():
                        _, aux_preds = torch.max(outputs_aux[labeled_aux_mask], 1)
                        correct_batch = (aux_preds == labels_aux[labeled_aux_mask]).sum().item()
                        aux_correct += correct_batch
                        total_labeled_aux += labeled_aux_mask.sum().item()

                # === 一致性损失（肿大数据）===
                if torch.any(unlabeled_mask):
                    # 学生模型预测 (强增强)
                    student_main_strong, _ = model(images_strong[unlabeled_mask])

                    # 一致性损失 (MSE)
                    consistency_loss = F.mse_loss(
                        F.softmax(student_main_strong, dim=1),
                        F.softmax(teacher_main[unlabeled_mask], dim=1)
                    )
                    losses.append(consistency_weight * consistency_loss)

                # 总损失
                loss = sum(losses)
                train_loss += loss.item()

                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 更新教师模型
                mean_teacher.update_teacher(alpha)

            # 学习率调整
            scheduler.step()
            warmup.step()

            # 训练统计
            train_loss /= len(train_loader)
            main_acc = main_correct / total_labeled_main if total_labeled_main > 0 else 0.0
            aux_acc = aux_correct / total_labeled_aux if total_labeled_aux > 0 else 0.0

            print(f"Train Epoch: {epoch + 1}, Loss: {train_loss:.4f}, Main Acc: {main_acc:.4f}, Aux Acc: {aux_acc:.4f}")

            # 5. 验证（使用教师模型）
            valid_loss, valid_main_acc = validate(
                mean_teacher.teacher,  # 使用教师模型验证
                valid_loader,
                device,
                loss_func_main
            )

            history['train_loss'].append(train_loss)
            history['valid_loss'].append(valid_loss)

            print(f"Val Epoch: {epoch + 1}, Loss: {valid_loss:.4f}, Acc: {valid_main_acc:.4f}")

            # 保存最佳模型
            if valid_main_acc > best_valid_acc:
                best_valid_acc = valid_main_acc
                best_epoch = epoch + 1
                torch.save(mean_teacher.teacher.state_dict(),
                           os.path.join(pt_dir, f'fold{fold_idx}-best-acc-model.pt'))
                print(f"New best model saved at epoch {best_epoch} with acc {best_valid_acc:.4f}")

            # 早停检查
            early_stopping(valid_loss, mean_teacher.teacher)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # 6. 保存训练曲线
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['valid_loss'], label='Valid Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.savefig(os.path.join(pt_dir, f'loss_curve_fold{fold_idx}.png'))
        plt.close()

        # 7. 测试最佳模型
        test_model = prepare_multi_task_model(
            category_num_main,
            category_num_aux,
            model_name,
            lr,
            num_epochs,
            device
        )[0]
        test_model.load_state_dict(torch.load(os.path.join(pt_dir, f'fold{fold_idx}-best-acc-model.pt')))
        test_model.to(device)

        test_acc, test_report = evaluate(
            test_model,
            test_loader,
            device,
            target_list_main
        )

        print(f"Test Accuracy: {test_acc:.4f}")
        print("Classification Report:")
        print(test_report)

        # 保存测试结果
        with open(os.path.join(pt_dir, f'test_report_fold{fold_idx}.txt'), 'w') as f:
            f.write(test_report)


# ================== 验证函数 ==================
def validate(model, loader, device, loss_func):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels_main = batch['label_main'].to(device)

            # 仅主任务
            outputs_main, _ = model(images)
            loss = loss_func(outputs_main, labels_main)

            total_loss += loss.item()
            _, preds = torch.max(outputs_main, 1)
            correct += torch.sum(preds == labels_main).item()
            total += labels_main.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total if total > 0 else 0.0

    return avg_loss, accuracy


# ================== 评估函数 ==================
def evaluate(model, loader, device, target_names):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            images = batch['image'].to(device)
            labels_main = batch['label_main'].to(device)

            outputs_main, _ = model(images)
            _, preds = torch.max(outputs_main, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_main.cpu().numpy())

    # 计算指标
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)

    print("Confusion Matrix:")
    print(cm)

    return accuracy, report


# ================== 分类任务入口 ==================
def classification():
    # 数据集配置
    data_dir = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集/'
    # 有标签数据 (1320例)
    labeled_txt_path = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/ori/20250702-良恶性2分类-all.txt'
    # 无标签数据 (5853例，其中肿大1067例)
    weak_label_txt_path = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/ori/20250812-肿大软标签.txt'
    # 目标类别
    target_list_main = [x.name for x in LymphPathologicCls2CN]  # 良恶性
    target_list_aux = [x.name for x in SwollenStatus]  # 肿大状态

    # 训练配置
    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'
    category_num_main = len(target_list_main)  # 2: 良性/恶性
    category_num_aux = len(target_list_aux)  # 2: 肿大/非肿大
    bs = 64  # 减小batch size以适应半监督
    lr = 0.0001
    patience = 30
    num_epochs = 500

    # 输出目录
    data = 'Section_QC/20250815-半监督-良恶性-肿大分类-'
    pt_dir = data + model_name + '-bs' + str(bs) + '-lr' + str(lr) + '/'
    os.makedirs(pt_dir, exist_ok=True)

    print(f'开始半监督训练: 主任务={category_num_main}分类, 辅任务={category_num_aux}分类')
    print(f'模型: {model_name}, batch size={bs}, 学习率={lr}')

    semi_supervised_train(
        data_dir=data_dir,
        labeled_txt_path=labeled_txt_path,
        weak_label_txt_path=weak_label_txt_path,
        num_epochs=num_epochs,
        bs=bs,
        pt_dir=pt_dir,
        category_num_main=category_num_main,
        category_num_aux=category_num_aux,
        model_name=model_name,
        device=device,
        lr=lr,
        target_list_main=target_list_main,
        target_list_aux=target_list_aux,
        patience=patience,
        alpha=0.995,  # EMA衰减率
        consistency_weight=5.0  # 一致性损失权重
    )
    print('训练完成')


if __name__ == '__main__':
    classification()
