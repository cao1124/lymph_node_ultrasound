#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound
@File    ：classify_txt.py
@IDE     ：PyCharm
@Author  ：cao xu
@Date    ：2025/6/19 上午10:18
"""
import os
from collections import Counter
from sklearn.utils import shuffle
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report
from classify_util import img_trans, prepare_model, EarlyStopping, BalanceDataSampler, DatasetTxt, func, LymphCls2CN
from torch.utils.data import DataLoader
import matplotlib
import matplotlib.pyplot as plt
import torch.multiprocessing
import warnings
matplotlib.use('AGG')
torch.multiprocessing.set_sharing_strategy('file_system')
warnings.filterwarnings("ignore")


def train(data_dir, txt_path, num_epochs, bs, pt_dir, category_num, model_name, device, lr, target_list, patience):
    # txt 5fold dataset
    with open(txt_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    temp = list(func(shuffle(lines, random_state=1), int(len(lines) * 0.2), m=5))
    test_list = [temp[4]]  # 固定测试集是第5折（index=4）
    for i in range(4):
        print('五折交叉验证 第{}次实验:'.format(i))
        train_list, valid_list = [], []
        for index, cross in enumerate(temp):
            if index == i:  # 验证集 = 当前第i折
                valid_list.append(cross)
            elif cross not in test_list:  # 训练集 = 剩下的数据（不包括测试集）
                train_list.append(cross)
        train_dataset = DatasetTxt(train_list, data_dir, transforms=img_trans['train'])
        valid_dataset = DatasetTxt(valid_list, data_dir, transforms=img_trans['valid'])
        test_dataset = DatasetTxt(test_list, data_dir, transforms=img_trans['valid'])
        train_size, valid_size, test_size = train_dataset.length, valid_dataset.length, test_dataset.length
        print('train_size:{}, valid_size:{}, test_size:{}'.format(train_size, valid_size, test_size))
        'class weight'
        label_count = Counter(train_dataset.labels)
        class_weights = []
        for cla in label_count:
            class_weights.append(1 / label_count.get(cla))
        'dataloader'
        sampler = BalanceDataSampler(train_dataset.labels)  # 创建采样器实例
        balanced_train_size = len(sampler)
        train_loader = DataLoader(train_dataset, bs, shuffle=False, num_workers=2, sampler=sampler)
        valid_loader = DataLoader(valid_dataset, bs, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, bs, shuffle=False, num_workers=2)
        'model, optimizer, scheduler, warmup, loss_function '
        model, optimizer, scheduler, warmup, loss_func = prepare_model(category_num, model_name, lr, num_epochs, device, weights=torch.tensor(class_weights))
        'EarlyStopping'
        early_stopping = EarlyStopping(pt_dir, patience=patience)
        best_test_acc, best_valid_acc, best_valid_recall, best_epoch = 0.0, 0.0, 0.0, 0
        history = []
        # error_sample = []
        for epoch in range(num_epochs):
            print("Epoch: {}/{}".format(epoch + 1, num_epochs))

            'train'
            train_loss = 0.0
            num_correct = 0
            model.train()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            for step, batch in enumerate(train_loader):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                # img_name = batch[2]

                outputs = model(inputs)
                # for r in range(len(torch.eq(outputs.argmax(dim=1), labels))):
                #     if torch.eq(outputs.argmax(dim=1), labels)[r].item() is False:
                #         error_sample.append(img_name[r] + ',' + str(labels[r].item())+'\n')
                loss_step = loss_func(outputs, labels)
                train_loss += loss_step.item()
                num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                optimizer.zero_grad()  # reset gradient
                loss_step.backward()
                optimizer.step()
                scheduler.step()
                warmup.step()

            train_loss /= len(train_loader)
            train_acc = num_correct / balanced_train_size
            print("Train Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, train_loss, train_acc))

            'validation'
            num_correct = 0
            valid_loss = 0.0
            valid_true, valid_pred = [], []
            model.eval()
            torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
            with torch.no_grad():
                for step, batch in enumerate(valid_loader):
                    inputs = batch[0].to(device)
                    labels = batch[1].to(device)
                    # img_name = batch[2]

                    outputs = model(inputs)
                    # for r in range(len(torch.eq(outputs.argmax(dim=1), labels))):
                    #     if torch.eq(outputs.argmax(dim=1), labels)[r].item() is False:
                    #         error_sample.append(img_name[r] + ',' + str(labels[r].item())+'\n')
                    loss_step = loss_func(outputs, labels)
                    valid_loss += loss_step.item()
                    num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                    valid_true.extend(labels.cpu().numpy())
                    valid_pred.extend(outputs.argmax(dim=1).cpu().numpy())

            valid_loss /= len(valid_loader)
            valid_acc = num_correct / valid_size
            print("Val Epoch: {}, Loss: {:.4f}, Acc: {:.4f}".format(epoch + 1, valid_loss, valid_acc))
            'best acc save checkpoint'
            if best_valid_acc < valid_acc:
                print('confusion_matrix:\n{}'.format(confusion_matrix(valid_true, valid_pred)))
                print('classification_report:\n{}'.format(classification_report(valid_true, valid_pred, target_names=target_list, digits=4)))
                best_valid_acc = valid_acc
                best_epoch = epoch + 1
                torch.save(model, pt_dir + 'fold' + str(i) + '-best-acc-model.pt')

            print("Epoch: {:03d}, Train Loss: {:.4f}, Acc: {:.4f}, Valid Loss: {:.4f}, Acc:{:.4f}" .format(epoch + 1, train_loss, train_acc, valid_loss, valid_acc))
            print("validation best: {:.4f} at epoch {}".format(best_valid_acc, best_epoch))
            # 早停止
            early_stopping(valid_loss, model)
            # 达到早停止条件时，early_stop会被置为True
            if early_stopping.early_stop:
                print("Early stopping")
                break  # 跳出迭代，结束训练
            history.append([train_loss, valid_loss])

        history = np.array(history)
        plt.clf()  # 清图
        plt.plot(history)
        plt.legend(['Tr Loss', 'Val Loss'])
        plt.xlabel('Epoch Number')
        plt.ylabel('Loss')
        plt.ylim(0, np.max(history))
        plt.savefig(pt_dir + 'loss_curve' + str(i) + '.png')

        'test'
        model = torch.load(pt_dir + 'fold' + str(i) + '-best-acc-model.pt')  # best-recall-model.pt
        num_correct = 0
        test_label, test_pred = [], []
        model.eval()
        torch.cuda.empty_cache()  # 释放缓存分配器当前持有的且未占用的缓存显存
        with torch.no_grad():
            for step, batch in enumerate(test_loader):
                inputs = batch[0].to(device)
                labels = batch[1].to(device)
                img_name = batch[2]
                outputs = model(inputs)
                'error sample'
                # for r in range(len(torch.eq(outputs.argmax(dim=1), labels))):
                #     if torch.eq(outputs.argmax(dim=1), labels)[r].item() is False:
                #         error_sample.append(img_name[r]+','+str(labels[r].item())+'\n')
                num_correct += torch.eq(outputs.argmax(dim=1), labels).sum().float().item()

                test_label.extend(labels.cpu().numpy())
                test_pred.extend(outputs.argmax(dim=1).cpu().numpy())

            test_acc = num_correct / test_size
            print("test accuracy: {:.4f}".format(test_acc))
            print('confusion_matrix:\n{}'.format(confusion_matrix(test_label, test_pred)))
            print('classification_report:\n{}'.format(classification_report(test_label, test_pred, target_names=target_list, digits=4)))
            # with open(f'{save_name}fold{i}-error-sample.txt', 'w') as f:
            #     f.writelines(error_sample)


def classification():
    # 甲状腺 左右侧叶峡部分类
    data_dir = '/mnt/disk1/caoxu/dataset/中山淋巴结/训练集/'
    txt_dir = '/mnt/disk1/caoxu/dataset/中山淋巴结/20250620-中山淋巴恶性瘤淋巴瘤2分类-补充训练.txt'
    data = 'Section_QC/20250620-中山淋巴恶性瘤淋巴瘤2分类-'
    target_list = [x.name for x in LymphCls2CN]

    os.environ['CUDA_VISIBLE_DEVICES'] = "3"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'resnet50'  # resnext50-bs128 densenet161-bs64 resnet18-bs256
    category_num = len(target_list)
    bs = 200
    lr = 0.0001
    patience = 100
    num_epochs = 1000
    pt_dir = data + model_name + '-bs' + str(bs) + '-lr' + str(lr) + '/'
    os.makedirs(pt_dir, exist_ok=True)
    print('测试{}-{}分类, 使用{}模型, batch size等于{}下的分类结果：'.format(data, category_num, model_name, bs))
    train(data_dir, txt_dir, num_epochs, bs, pt_dir, category_num, model_name, device, lr, target_list, patience)
    print('done')


if __name__ == '__main__':
    classification()
