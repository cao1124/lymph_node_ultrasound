#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Training script for "第三部分细分-转移6分类" with dual-image input:
    input #1: original big image
    input #2: ROI crop image (path by .replace('...对比', '...对比-crop'))
The rest of the training/eval structure follows the user's classify_txt.py.
"""
import os
from collections import Counter
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report

import torch
import torch.multiprocessing
import warnings
import matplotlib
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np

from classify_util import img_trans, EarlyStopping, BalanceDataSampler, TransCls6CN
from classify_util_fusion import DatasetTxtTwo, prepare_model_fusion
from fusion_cross_shared import prepare_model_fusion_cross
matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def train(data_dir, txt_path, num_epochs, bs, pt_dir, category_num, device, lr, target_list, patience):
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    temp = list(func(shuffle(lines, random_state=1), int(len(lines) * 0.2), m=5))
    test_list = [temp[4]]  # 固定测试集为第5折

    for i in range(4):
        print('五折交叉验证 第{}次实验:'.format(i))
        train_list, valid_list = [], []
        for index, cross in enumerate(temp):
            if index == i:
                valid_list.append(cross)
            elif cross not in test_list:
                train_list.append(cross)

        # 两图输入的数据集
        train_dataset = DatasetTxtTwo(train_list, data_dir, transforms=img_trans['train'])
        valid_dataset = DatasetTxtTwo(valid_list, data_dir, transforms=img_trans['valid'])
        test_dataset  = DatasetTxtTwo(test_list,  data_dir, transforms=img_trans['valid'])

        train_size, valid_size, test_size = train_dataset.length, valid_dataset.length, test_dataset.length
        print('train_size:{}, valid_size:{}, test_size:{}'.format(train_size, valid_size, test_size))

        # 类别权重（与原脚本一致）
        label_count = Counter(train_dataset.labels)
        class_weights = []
        for cla in label_count:
            class_weights.append(1 / label_count.get(cla))

        # dataloader
        sampler = BalanceDataSampler(train_dataset.labels)
        balanced_train_size = len(sampler)
        train_loader = DataLoader(train_dataset, bs, shuffle=False, num_workers=2, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, bs, shuffle=False, num_workers=2)
        test_loader  = DataLoader(test_dataset,  bs, shuffle=False, num_workers=2)

        # 模型 + 优化器 + 调度器 + warmup + 损失
        # model, optimizer, scheduler, warmup, loss_func = prepare_model_fusion(
        #     category_num=category_num, lr=lr, num_epochs=num_epochs, device=device,
        #     weights=torch.tensor(class_weights), backbone="resnet50"
        # )

        # fusion cross shared
        model, optimizer, scheduler, warmup, loss_func = prepare_model_fusion_cross(
            num_classes=category_num,
            lr=lr,
            num_epochs=num_epochs,
            device=device,
            class_weights=torch.tensor(class_weights),
            pretrained=True
        )

        early_stopping = EarlyStopping(pt_dir, patience=patience)
        best_valid_acc, best_epoch = 0.0, 0
        history = []

        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))

            # --------------------- Train ---------------------
            train_loss = 0.0
            num_correct = 0
            model.train()
            torch.cuda.empty_cache()
            for step, batch in enumerate(train_loader):
                # batch[0] is a tuple: (big_batch, crop_batch)
                inputs = batch[0]
                labels = batch[1].to(device)

                # move both images to device
                big = inputs[0].to(device)
                crop = inputs[1].to(device)

                outputs = model((big, crop))
                loss_step = loss_func(outputs, labels)
                train_loss += loss_step.item()
                num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                optimizer.zero_grad()
                loss_step.backward()
                optimizer.step()
                scheduler.step()
                warmup.step()

            train_loss /= len(train_loader)
            train_acc = num_correct / balanced_train_size
            print("Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, train_loss, train_acc))

            # ------------------- Validation ------------------
            num_correct = 0
            valid_loss = 0.0
            valid_true, valid_pred = [], []
            model.eval()
            torch.cuda.empty_cache()
            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    inputs = batch[0]
                    labels = batch[1].to(device)

                    big = inputs[0].to(device)
                    crop = inputs[1].to(device)

                    outputs = model((big, crop))
                    loss_step = loss_func(outputs, labels)
                    valid_loss += loss_step.item()
                    num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                    valid_true.extend(labels.cpu().numpy())
                    valid_pred.extend(outputs.argmax(dim=1).cpu().numpy())

            valid_loss /= len(valid_loader)
            valid_acc = num_correct / valid_size
            print("Val Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))

            if best_valid_acc < valid_acc:
                from sklearn.metrics import confusion_matrix, classification_report
                print('confusion_matrix:\n{}'.format(confusion_matrix(valid_true, valid_pred)))
                print('classification_report:\n{}'.format(classification_report(valid_true, valid_pred, target_names=target_list, digits=4)))
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                torch.save(model, pt_dir + 'fold' + str(i) + '-best-acc-model.pt')

            print("Epoch: {:03d}, Train Loss: {:.4f}, Acc: {:.4f}, Valid Loss: {:.4f}, Acc:{:.4f}"
                  .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
            print("validation best: {:.4f} at epoch {}".format(best_valid_acc, best_epoch))

            early_stopping(valid_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            history.append([train_loss, valid_loss])

        history = np.array(history)
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(history)
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0, np.max(history))
        os.makedirs(pt_dir, exist_ok=True)
        plt.savefig(pt_dir + 'loss_curve' + str(i) + '.png')

        # ---------------------- Test -----------------------
        model = torch.load(pt_dir + 'fold' + str(i) + '-best-acc-model.pt')
        num_correct = 0
        test_label, test_pred = [], []
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs = batch[0]
                labels = batch[1].to(device)
                img_name = batch[2]

                big = inputs[0].to(device)
                crop = inputs[1].to(device)
                outputs = model((big, crop))

                num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()
                test_label.extend(labels.cpu().numpy())
                test_pred.extend(outputs.argmax(dim=1).cpu().numpy())

        test_acc = num_correct / test_size
        print("test accuracy: {:.4f}".format(test_acc))
        print('confusion_matrix:\n{}'.format(confusion_matrix(test_label, test_pred)))
        print('classification_report:\n{}'.format(classification_report(test_label, test_pred, target_names=target_list, digits=4)))


def func(listTemp, n, m=5):
    """Split list for 5-fold as in original code."""
    count = 0
    for i in range(0, len(listTemp), n):
        count += 1
        if count == m:
            yield listTemp[i:]
            break
        else:
            yield listTemp[i:i + n]


def classification():
    # 20251016-第三部分细分-转移6分类
    data_dir = ''
    txt_dir = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/20251016-第三部分细分/20251016-第三部分细分-转移6分类.txt'
    data = 'Section_QC/20251016-第三部分细分-转移6分类-'
    target_list = [x.name for x in TransCls6CN]

    os.environ['CUDA_VISIBLE_DEVICES'] = "6"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    category_num = len(target_list)
    bs = 64
    lr = 0.0001
    patience = 100
    num_epochs = 1000
    pt_dir = data + 'resnet50-fusion' + '-bs' + str(bs) + '-lr' + str(lr) + '/'
    os.makedirs(pt_dir, exist_ok=True)
    print('测试{}-{}分类, 使用{}模型, batch size等于{}下的分类结果：'.format(data, category_num, 'resnet50-fusion', bs))
    train(data_dir, txt_dir, num_epochs, bs, pt_dir, category_num, device, lr, target_list, patience)
    print('done')


if __name__ == '__main__':
    classification()
