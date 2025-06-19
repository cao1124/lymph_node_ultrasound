import argparse
import os
import cv2
import sys

import matplotlib.pyplot as plt
import wandb
import math
import pandas as pd
from tqdm import tqdm

import torch
from torchnet import meter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torch import nn
from torch.nn import *
import torchvision.utils as vutils
from torch.cuda.amp import autocast as autocast
from torch.optim import *
from torch.autograd import *
import torchvision.models as models
from warmup_scheduler import GradualWarmupScheduler
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import classification_report,confusion_matrix

from trainers.base_trainer import BaseTrainer
from utils.mixup import mixup_data, mixup_criterion
from utils.loss import LabelSmoothingCrossEntropy, softmax, FocalLoss, Weighted_Focal_Loss, HardSampleLoss
from util import compute_metrics, read_json
import backbone.convnext as convnext
from backbone.noisyResnet import nnResnet_50_default as nnR5D
from postprocess.plot_util import plot_auc
from dataset import datasets_dict, ZS_model_checkpoint_dict
from typing import Any

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))

class Trainer(BaseTrainer):

    def __init__(self, args: argparse.Namespace):
        super().__init__(args)
        labels= [i.strip() for i in open(args.labels_path).readlines()]
        self.args.num_classes=len(labels)
        self.label2id = dict(zip(labels,list(range(len(labels)))))
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))

        if hasattr(args, 'split_image_file'):
            if not isinstance(args.split_image_file, list):
                self.split_image = read_json(args.split_image_file)
            else:
                assert len(args.split_image_file) == len(args.domain_dir)
                self.split_image = [read_json(_file_path) for _file_path in args.split_image_file]

        self.metric = ''
        if hasattr(args, 'split_patient_file'):
            self.split_patient = read_json(args.split_patient_file)
        if hasattr(args, 'boxes_path'):
            self.boxes_path = args.boxes_path
        self.metric_method = args.metric_method

    def train(self):
        args = self.args

        project_name = args.CLS + getattr(args, 'hospital_name', '')
        run_name = args.model_name + '_' + args.running_name
        wandb.init(
            project=project_name,
            name=run_name,
            config=vars(args)
        )
        self.logger.info("Model name: %s" % args.model_name)
        self.logger.info("domain dir: %s" % args.domain_dir)
        if hasattr(args, 'domain_list'):
            self.logger.info("domain list: %s" % args.domain_list)
        if hasattr(args, 'split_image_file'):
            self.logger.info("split image file: %s" % args.split_image_file)
        #* read datasets
        self.logger.info("加载域内数据的训练集...")
        train_dataset = self._get_datasets(dataset_type='train', ZhongShan=True)
        self.logger.info("加载域内数据的验证集...")
        val_dataset = self._get_datasets(dataset_type='val', ZhongShan=True)
        self.logger.info("数据加载成功\n")
        self.logger.info("训练集样本数量: %s" % train_dataset.length)
        self.logger.info("验证集样本数量: %s" % val_dataset.length)

        train_data_num = train_dataset.length
        epoch_iteration = math.ceil(train_data_num / (args.train_batch_size * self.args.ngpu))
        self.logger.info("总的训练epoch数量: %s" % args.epochs)
        self.logger.info("每个epoch的训练batch数量: %s" % epoch_iteration)
        self.logger.info("开始训练\n")

        #* load model, get optimizer, get criterion, get scheduler
        if args.resume_train:
            model, optimizer, scheduler, stop_epoch = self._recover_from(args.resume_dir)
            criterion = self._get_criterion(args.criterion_name)
        else:
            model = self._load_model(model_name=args.model_name, n_class=args.num_classes)
            # model = self._resetBias(model)

            optimizer = self._get_optimizer(args.optimizer_name, args.lr, model)
            criterion = self._get_criterion(args.criterion_name)
            # lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch_iteration)
            scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=lr_scheduler)
            stop_epoch = -1

        if args.ngpu > 1:
            device_ids = [int(id) for id in args.device_ids.split(',')]
            assert args.ngpu == len(device_ids) and int(args.device_id) in device_ids
            model = torch.nn.DataParallel(model, device_ids=device_ids)

        #* train
        for epoch in range(stop_epoch+1, args.epochs):
            train_loss, gts, pred_s= self._train_epoch(model, criterion, optimizer, scheduler, train_dataset, epoch_iteration, epoch, args.mixup, train_analyse=True)
         
            # compute trian metrics

            train_metric = compute_metrics(gts, pred_s)
            wandb.log({
                'train_malign_recall': train_metric['sensitivity'],
                'train_benign_recall': train_metric['specificity'],
                'train_acc': train_metric['benign_malig_acc']
            })


            # * eval train sets and validation sets
            if (epoch - 1) % 100 == 0:
                with open (self.metric_file_path, 'a') as fs:
                    print("评估训练集", file=fs)
                self.logger.info("评估训练集")
                train_metric = self._eval(model, criterion, train_dataset, epoch, record_loss=False)

                train_malign_recall = train_metric["sensitivity"]
                train_benign_recall = train_metric['specificity']
                train_benign_malign_acc = train_metric['benign_malig_acc']

                if args.use_tensorboard:
                    self.writer.add_scalar(self.metric_method, float(train_metric[self.metric_method]), epoch)


            with open (self.metric_file_path,'a') as fs:
                print("评估验证集",file=fs)
            self.logger.info("评估验证集")
            eval_loss, eval_metric = self._eval(model, criterion, val_dataset, epoch, record_loss=True)

            eval_malign_recall = eval_metric['sensitivity']
            eval_benign_recall = eval_metric['specificity']
            eval_benign_malign_acc = eval_metric['benign_malig_acc']

            # wandb write data
            wandb.log(
                {
                    'train_loss': train_loss,
                    'test_loss': eval_loss})

            if args.eval_trainset:
                wandb.log({
                    'train_malign_recall': train_malign_recall,
                    'test_malign_recall': eval_malign_recall,
                })

                wandb.log({
                    'train_benign_recall': train_benign_recall,
                    'test_benign_recall': eval_benign_recall,
                })

                wandb.log({
                    'train_acc':train_benign_malign_acc,
                    'test_acc':eval_benign_malign_acc
                })

            else:
                wandb.log({
                    'test_malign_recall': eval_malign_recall,
                    'test_benign_recall': eval_benign_recall,
                    'test_acc':eval_benign_malign_acc
                })
            if args.use_tensorboard:
                self.writer.add_scalar(self.metric_method, float(eval_metric[self.eval_metric]), epoch)

            self.save_top_recall_model(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                malign_performance=eval_malign_recall,
                benign_performance=eval_benign_recall,
                epoch=epoch,
                num=4
            )

            self.save_topK(num=args.save_model_num, model=model, optimizer=optimizer, scheduler=None, performace=eval_metric["weighted_f1_score"],
                           epoch=epoch)


        self.logger.info('\n')
        self.logger.info(f"训练过程中最好的{self.eval_metric}为:{self.best_performace}")
        self.logger.info("Logged in: %s" % self.log_path)
        self.logger.info("Saved in: %s" % self.save_path)


    # def test(self):
    #     args = self.args
    #     if hasattr(args, 'split_image_file'):
    #         self.split_image = read_json(args.split_image_file)
    #
    #     self.logger.info("model path: %s" % args.model_path)
    #     self.logger.info("domain dir: %s" % args.domain_dir)
    #
    #     notList = not isinstance(args.domain_list, list)
    #
    #     if notList:
    #         if args.domain_list.endswith('.txt'):
    #             domain_list = open(args.domain_list).read().split('\n')
    #         else:
    #             domain_list = args.domain_list.split('_') if len(args.domain_list) > 0 else []
    #     else:
    #         domain_list = args.domain_list
    #
    #     self.logger.info("domain list: %s" % domain_list)
    #     out_domain_list = args.out_domain_list.split('_') if len(args.out_domain_list) > 0 else []
    #     self.logger.info("out domain list: %s" % out_domain_list)
    #     all_domain_list = []
    #     if len(domain_list) > 0:
    #         if len(domain_list) == 1:
    #             all_domain_list.extend(domain_list)
    #         else:
    #             all_domain_list.extend(domain_list + ['域内平均', '整体计算'])
    #
    #     if len(out_domain_list) > 0:
    #         all_domain_list.extend(out_domain_list + ['域外平均', '整体计算'])
    #
    #     metric_result = {
    #         "specificity": [],
    #         "sensitivity": [],
    #         "weighted_f1_score": [],
    #         "benign_malig_acc": [],
    #         "acc":[],
    #     }
    #
    #     # load model
    #     model = torch.load(self.args.model_path, map_location=self.device)
    #     if isinstance(model, nn.DataParallel):
    #         model = model.module
    #     if len(domain_list) > 0:
    #         for in_domain_name in domain_list:
    #             val_dataset = DomainDatasetTest(
    #                 domain_dir=args.domain_dir,
    #                 domain_list=[in_domain_name],
    #                 split_data=self.split_image,
    #                 dataset_type=args.dataset_type,
    #                 img_size=args.target_size,
    #                 is_rgb=args.is_rgb,
    #             )
    #             self.logger.info("[%s]测试样本数量: %s" % (in_domain_name, val_dataset.length))
    #             # evaluate
    #             with open(self.metric_file_path, 'a') as fs:
    #                 print(f"当前数据集：{in_domain_name}", file=fs)
    #             metric = self._eval(model, None, val_dataset, None, False)
    #             for metric_type in metric_result:
    #                 metric_result[metric_type].append(metric[metric_type])
    #
    #         if len(domain_list) != 1:
    #             # 计算宏平均（即域外平均）
    #             # macro avg = (domain_1_avg + domain_2_avg + domain_3_avg + ......) / len(domain)
    #             for metric_type in metric_result:
    #                 metric_result[metric_type].append(
    #                     round((sum(metric_result[metric_type])) / (len(metric_result[metric_type])), 4))
    #             # 计算微平均（即整体计算）
    #             # 把所有domain的数据当作一个整体，计算TP,FN,FP,TN, 然后统计各个指标
    #             val_dataset = DomainDatasetTest(
    #                 domain_dir=args.domain_dir,
    #                 domain_list=domain_list,
    #                 split_data=self.split_image,
    #                 dataset_type=args.dataset_type,
    #                 img_size=args.target_size,
    #                 is_rgb=args.is_rgb,
    #             )
    #             self.logger.info("总计测试样本数量: %s" % (val_dataset.length))
    #             # evaluate
    #             with open(self.metric_file_path, 'a') as fs:
    #                 print(f"整体计算", file=fs)
    #             metric = self._eval(model, None, val_dataset, None, False)
    #             for metric_type in metric_result:
    #                 metric_result[metric_type].append(metric[metric_type])
    #
    #
    #     if len(out_domain_list) > 0:
    #         for out_domain_name in out_domain_list:
    #             val_dataset = DomainDatasetTest(
    #                 domain_dir=args.domain_dir,
    #                 domain_list=[out_domain_name],
    #                 split_data=self.split_image,
    #                 dataset_type=args.dataset_type,
    #                 img_size=args.target_size,
    #                 is_rgb=args.is_rgb,
    #             )
    #             self.logger.info("[%s]测试样本数量: %s" % (out_domain_name, val_dataset.length))
    #             # evaluate
    #             with open(self.metric_file_path, 'a') as fs:
    #                 print(f"当前数据集：{out_domain_name}", file=fs)
    #             metric = self._eval(model, None, val_dataset, None, False)
    #             for metric_type in metric_result:
    #                 metric_result[metric_type].append(metric[metric_type])
    #
    #         outdomain_num = len(out_domain_list)
    #         for metric_type in metric_result:
    #             metric_result[metric_type].append(round((sum(metric_result[metric_type][-outdomain_num:])) / (
    #                 len(metric_result[metric_type][-outdomain_num:])), 4))
    #
    #         val_dataset = DomainDatasetTest(
    #             domain_dir=args.domain_dir,
    #             domain_list=out_domain_list,
    #             split_data=self.split_image,
    #             dataset_type=args.dataset_type,
    #             img_size=args.target_size,
    #             is_rgb=args.is_rgb,
    #         )
    #         self.logger.info("总计测试样本数量: %s" % (val_dataset.length))
    #         # evaluate
    #         with open(self.metric_file_path, 'a') as fs:
    #             print(f"整体计算", file=fs)
    #         metric = self._eval(model, None, val_dataset, None, False)
    #         for metric_type in metric_result:
    #             metric_result[metric_type].append(metric[metric_type])
    #
    #     df = pd.DataFrame({'domain': all_domain_list,
    #                        '良性-recall': metric_result["specificity"],
    #                        '恶性-recall': metric_result["sensitivity"],
    #                        "wei_f1": metric_result["weighted_f1_score"],
    #                        'benign_malig_acc': metric_result["benign_malig_acc"],
    #                        'acc':metric_result["acc"]
    #                        })
    #     writer = pd.ExcelWriter(os.path.join(self.log_path, 'val_res.xlsx'))
    #     df.to_excel(writer, 'val_res', index=False)
    #     writer._save()
    #
    #     self.logger.info("Logged in: %s" % self.log_path)


    def test_pkl(self):
        # 读取pkl文件的数据
        args = self.args

        boxes_path = args.patch_boxes_path
        self.logger.info("model path: %s" % args.model_path)
        self.logger.info("domain dir: %s" % args.domain_dir)

        notList = not isinstance(args.domain_list, list)

        if notList:
            if args.domain_list.endswith('.txt'):
                domain_list = open(args.domain_list).read().split('\n')
            else:
                domain_list = args.domain_list.split('_') if len(args.domain_list) > 0 else []
        else:
            domain_list = args.domain_list

        self.logger.info("domain list: %s" % domain_list)
        out_domain_list = args.out_domain_list.split('_') if len(args.out_domain_list) > 0 else []
        self.logger.info("out domain list: %s" % out_domain_list)
        all_domain_list = []
        if len(domain_list) > 0:
            if len(domain_list) == 1:
                all_domain_list.extend(domain_list)
            else:
                all_domain_list.extend(domain_list + ['域内平均', '整体计算'])

        if len(out_domain_list) > 0:
            all_domain_list.extend(out_domain_list + ['域外平均', '整体计算'])

        metric_result = {
            "specificity": [],
            "sensitivity": [],
            "weighted_f1_score": [],
            "benign_malig_acc": [],
            "acc":[],
        }

        # load model
        UseDatasets = datasets_dict[self.args.dataset_name]

        # model = torch.load(self.args.model_path, map_location=self.device)
        model_path = model_checkpoint_dict[self.args.dataset_name]
        model = torch.load(model_path, map_location=self.device)
        if isinstance(model, nn.DataParallel):
            model = model.module
        if len(domain_list) > 0:
            for domain_name in domain_list:
                test_datasets = UseDatasets(
                    domain_dir=args.domain_dir,
                    domain_list=[domain_name],
                    boxes_path=boxes_path,
                    dataset_type='test',
                    patch_size=args.target_size,
                    crop_type='',
                    is_rgb=args.is_rgb,
                    OutDomainTest=True
                )

                self.logger.info("[%s]测试样本数量: %s" % (domain_name, test_datasets.length))
                # evaluate
                with open(self.metric_file_path, 'a') as fs:
                    print(f"当前数据集：{domain_name}", file=fs)
                metric = self._eval(model, None, test_datasets, None, False)
                for metric_type in metric_result:
                    metric_result[metric_type].append(metric[metric_type])

            if len(domain_list) != 1:
                # 计算宏平均（即域外平均）
                for metric_type in metric_result:
                    metric_result[metric_type].append(
                        round((sum(metric_result[metric_type])) / (len(metric_result[metric_type])), 4))

                # 计算域外平均
                test_datasets = UseDatasets(
                    domain_dir=args.domain_dir,
                    domain_list=domain_list,
                    boxes_path=boxes_path,
                    dataset_type='test',
                    patch_size=args.target_size,
                    crop_type='',
                    is_rgb=args.is_rgb,
                    OutDomainTest=True
                )
                self.logger.info("总计测试样本数量: %s" % (test_datasets.length))
                with open(self.metric_file_path, 'a') as fs:
                    print(f"整体计算", file=fs)
                metric = self._eval(model, None, test_datasets, None, False)
                for metric_type in metric_result:
                    metric_result[metric_type].append(metric[metric_type])

        df = pd.DataFrame({'domain': all_domain_list,
                           '良性-recall': metric_result["specificity"],
                           '恶性-recall': metric_result["sensitivity"],
                           "wei_f1": metric_result["weighted_f1_score"],
                           'benign_malig_acc': metric_result["benign_malig_acc"],
                           'acc': metric_result["acc"]
                           })
        writer = pd.ExcelWriter(os.path.join(self.log_path, 'phy_val_res.xlsx'))
        df.to_excel(writer, 'val_res', index=False)
        writer._save()
        self.logger.info("Logged in: %s" % self.log_path)


    # 在训练时去除偏置
    def _resetBias(self, model):
        for name, module in model.named_modules():
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    module.bias = None
        return model


    def _checkBias(self, model):
        for param, module in model.named_modules():
            if hasattr(module, 'bias'):
                if module.bias is not None:
                    print(module)
                    print(module.bias)
                    print('模型有偏置, 结束训练')
                    sys.exit()

    def _load_model(self, model_name, n_class, pretrained=False):
        # cnn-based
        if model_name == "vgg16_bn_pretrain":
            model = models.vgg16_bn(pretrained=pretrained, num_classes=n_class)
        elif model_name == "vgg16_bn":
            model = models.vgg16_bn(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnet18_pretrain":
            model = models.resnet18(pretrained=True, num_classes=n_class)
        elif model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnet34_pretrain":
            model = models.resnet34(pretrained=True, num_classes=n_class)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnet50_pretrain":
            model = models.resnet50(pretrained=True, num_classes=n_class)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnet101_pretrain":
            model = models.resnet101(pretrained=True, num_classes=n_class)
        elif model_name == "resnet101":
            model = models.resnet101(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnet152_pretrain":
            model = models.resnet152(pretrained=pretrained)
            model.fc = nn.Linear(in_features=2048, out_features=n_class, bias=True)
        elif model_name == "densenet121_pretrain":
            model = models.densenet121(pretrained=True, num_classes=n_class)
        elif model_name == "densenet121":
            model = models.densenet121(pretrained=pretrained, num_classes=n_class)
        elif model_name == "densenet161_pretrain":
            model = models.densenet161(pretrained=True)
            model.classifier = nn.Linear(2208, n_class, bias=False)
        elif model_name == "densenet161":
            model = models.densenet161(pretrained=pretrained, num_classes=n_class)
            model.classifier = nn.Linear(2208, n_class, bias=True)
        elif model_name == "densenet201_pretrain":
            model = models.densenet201(pretrained=True, num_classes=n_class)
        elif model_name == "densenet201":
            model = models.densenet201(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnext101_32x8d_pretrain":
            model = models.resnext101_32x8d(pretrained=pretrained, num_classes=n_class)
        elif model_name == "resnext101_32x8d":
            model = models.resnext101_32x8d(pretrained=pretrained, num_classes=n_class)
        elif model_name == "efficientnet-b0":
            model = EfficientNet.from_pretrained('efficientnet-b0')
            model._fc = nn.Linear(1280, n_class)
        elif model_name == "convnext_large":
            model = convnext.convnext_large(pretrained=pretrained)
            model.head = nn.Linear(in_features=1536, out_features=n_class, bias=True)
        # transformer_based
        elif model_name == 'vit_b_16':
            model = models.vit_b_16(pretrained=pretrained, num_classes=n_class)
        elif model_name == 'vit_b_32':
            model = models.vit_b_32(pretrained=pretrained, num_classes=n_class)
        elif model_name == 'swin':
            model = models.swin_b(pretrained=pretrained, num_classes=n_class)
        elif model_name == 'max_vit':
            model = models.maxvit_t(pretrained=pretrained, num_classes=n_class)
        elif model_name == 'nnresnet50':
            model = nnR5D(pretrained=pretrained, num_classes=n_class)
        # LLM_based
        elif model_name == '':
            ...
        else:
            raise NameError(f'暂不支持该模型: {model_name}')
        model.to(self.device)
        return model


    def _recover_from(self, resume_dir):
        model = torch.load(os.path.join(resume_dir, 'model.pt'))# 加载模型module
        # 挂载到指定设备上
        model = model.to(self.device)
        optimizer = torch.load(os.path.join(resume_dir, 'extra.state'), map_location=self.device)['optimizer']  
        scheduler = torch.load(os.path.join(resume_dir, 'extra.state'), map_location=self.device)['scheduler'] 
        epoch = torch.load(os.path.join(resume_dir, 'extra.state'))['epoch'] 
        return model, optimizer, scheduler, int(epoch)


    def _get_optimizer(self, optimizer_name, lr, model):
        if optimizer_name == 'Adam':
            if self.args.use_weight_decay:
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                betas=(0.9, 0.99),
                                weight_decay=self.args.weight_decay
                                )
            else:
                optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=lr,
                                betas=(0.9, 0.99),
                                )
        elif optimizer_name == "AdamW":
            optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr,
                              )
        elif optimizer_name == "SGD":
            optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=lr,
                              ) 
        else:
            raise NameError(f'暂不支持该优化器: {optimizer_name}')
        return optimizer

    def _get_datasets(self, dataset_type, **kwargs):
        args = self.args
        UseBaseData = datasets_dict[self.args.dataset_name]
        if dataset_type == 'train':
            UseDatasets = UseBaseData(
                domain_dir=args.domain_dir,
                domain_list=None,
                boxes_path=args.patch_boxes_path,
                dataset_type='train',
                patch_size=args.target_size,
                crop_type=args.crop_type,
                is_rgb=args.is_rgb,
                aug_prob=args.aug_prob,
                gain_aug_prob=args.gain_aug_prob,
                gain_range=args.gain_range,
                **kwargs
            )
        elif dataset_type == 'test' or dataset_type == 'val':
            UseDatasets = UseBaseData(
                domain_dir=args.domain_dir,
                domain_list=None,
                boxes_path=args.patch_boxes_path,
                dataset_type=dataset_type,
                patch_size=args.target_size,
                crop_type=args.crop_type,
                is_rgb=args.is_rgb,
                **kwargs
            )
        else:
            raise NameError(f'暂不支持该数据集类型: {dataset_type}')
        return  UseDatasets

    def _get_criterion(self, criterion_name):
        if criterion_name == 'CrossEntropy':
            criterion = CrossEntropyLoss()
        elif criterion_name == "LabelSmoothingCrossEntropy":
            criterion = LabelSmoothingCrossEntropy()
        elif criterion_name == 'Weighted_CrossEntropy':
            criterion = CrossEntropyLoss(weight=torch.tensor([2.5, 1.0]).to(self.device))
        elif criterion_name == 'FocalLoss':
            criterion = FocalLoss()
        elif criterion_name == 'Weighted_Focal':
            criterion = Weighted_Focal_Loss(weight=torch.Tensor([2.5, 1.0]).to(self.device))
        elif criterion_name == 'HardSampleLoss':
            criterion = HardSampleLoss(threshold=0.7, weight_type='exp')
        else:
            raise NameError(f'暂不支持该损失函数: {criterion_name}')
        return criterion                  
                         

    def _train_epoch(self, 
                     model: torch.nn.Module, 
                     criterion: None, 
                     optimizer: None, 
                     scheduler: None, 
                     train_dataset: None,
                     epoch_iteration: int, 
                     epoch: int, mixup: None, train_analyse:bool=False):
        
        self.logger.info("Train epoch: %s" % epoch)
        with open (self.metric_file_path,'a') as fs:
            print("Train epoch: %s" % epoch, file=fs)

        #* create data loader
        data_loader = DataLoader(train_dataset,
                                self.args.train_batch_size * self.args.ngpu,
                                shuffle=True,
                                num_workers=8,
                                pin_memory=False,
                                )
     
        iteration = 0
        loss_meter = meter.AverageValueMeter()
        lr_meter = meter.AverageValueMeter()
        model.train()
        flag =True

        # 用来存放训练过程中模型的拟合情况, 定位具体的问题
        gts = []
        pred_s = []

        for batch in tqdm(data_loader, total=epoch_iteration, desc='Train epoch %s' % epoch):

            inputs = Variable(batch[0].to(self.device))
            labels = Variable(batch[1].to(self.device))

            # while epoch < 2 and flag:
            #     os.makedirs(os.path.join(self.log_path, 'train_images'), exist_ok=True)
            #     flag = False
            #     images = batch[2].cpu().numpy()[:8]
            #     for idx, single_image in enumerate(images):
            #         cv2.imwrite(os.path.join(self.log_path, 'train_images', f'train_epoch_{epoch}_image_{idx}.png'), single_image)
            #     batch_images = (batch[2].float() / 255.0).permute(0, 3, 1, 2)
            #     grid = vutils.make_grid(batch_images, nrow=8, normalize=True, scale_each=True)
            #     grid = (grid * 255).byte()
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
            #     plt.axis('off')
            #     plt.savefig(os.path.join(self.log_path, 'train_images', f'train_epoch_{epoch}.png'))
            #     plt.close()

            optimizer.zero_grad()
            if mixup == True:
                inputs, targets_a, targets_b, lam = mixup_data(inputs, labels, 1, True, self.device)
                inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
                # with autocast():
                outputs = model(inputs)
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                loss_step = loss_func(criterion, outputs)
            else:
                outputs = model(inputs)
                loss_step = criterion(outputs, labels)

            gts.extend(labels.cpu().numpy())
            pred_s.extend(outputs.argmax(dim=1).cpu().numpy())

            loss_meter.add(float(loss_step))
            lr = self._get_lr(optimizer)[0]
            lr_meter.add(lr)

            #* 每一个iteration都记录损失和学习率
            if self.args.use_tensorboard:
                cur_iteration = epoch_iteration * epoch + iteration
                self.writer.add_scalar("train_loss", loss_meter.value()[0], cur_iteration)
                self.writer.add_scalar("lr", lr, cur_iteration)

            loss_step.backward()
            optimizer.step()
            iteration += 1
        scheduler.step()  
        return loss_meter.value()[0], gts, pred_s if train_analyse else loss_meter.value()[0]

    def test(self):

        args = self.args
        name = args.model_name
        model_weights = ZS_model_checkpoint_dict[name]

        trained_model = torch.load(model_weights, map_location=self.device)
        test_datasets = self._get_datasets(dataset_type='test', ZhongShan=True)
        metric_dict = self._eval(trained_model, None, test_datasets, None)

        gt = metric_dict['gt']
        prob_confidence = metric_dict['prob_confidence']
        save_path = os.path.join(self.log_path, 'auc_images')
        os.makedirs(save_path, exist_ok=True)
        plot_auc(gt, prob_confidence, title=f'roc_{name}', save_path=f'{save_path}/{name}_roc.png')
        return metric_dict



    def _eval(self,
              model: torch.nn.Module,
              criterion: Any,
              dataset: Any, epoch: Any,
              record_loss:bool=False):
        self.logger.info("Evaluate result:")

        if isinstance(model, DataParallel):
            # currently no multi GPU support during evaluation
            model = model.module
        img_paths_all, confidence = [],[]
        # create evaluaotor
        data_loader = DataLoader(dataset,
                                self.args.eval_batch_size * self.args.ngpu,
                                shuffle=False,
                                num_workers=4
                                )
        loss_meter = meter.AverageValueMeter()
        with torch.no_grad():
            model.eval()

            # iterate batches
            total = math.ceil(dataset.length / (self.args.eval_batch_size * self.args.ngpu))
            true_labels = []
            pred_labels = []
            pred_positive_probs = []
            if epoch is None:
                desc = 'Evaluating'
            else:
                desc = 'Evaluate epoch %s' % epoch 
            for batch in tqdm(data_loader, total=total, desc=desc):
                
                inputs = Variable(batch[0].to(self.device))
                labels = Variable(batch[1].to(self.device))
                # img_paths = batch[2]
                # img_paths_all.extend(img_paths)
                # inputs = Variable(batch[0].cuda())
                # labels = Variable(batch[1].cuda())
              
                outputs = model(inputs)
                if criterion is not None:
                    loss = criterion(outputs, labels)
                    loss_meter.add(float(loss.cpu().numpy()))

                pred = outputs.argmax(dim=1)
                outputs_softmax = softmax(outputs)
                # 方便sklearn计算相应的指标
                labels_b = labels.cpu().numpy()
                preds_b = pred.cpu().numpy()
                true_labels.extend(labels_b)
                pred_labels.extend(preds_b)
                for i in range(len(labels_b)):
                    pred_positive_probs.append(outputs_softmax[i][1].cpu().item())
                    confidence.append(round(outputs_softmax[i][preds_b[i]].cpu().item(), 5))
          
        #把数字标签转回汉字，便于观察分析
        true_labels_chinese=[]
        pred_labels_chinese=[]
        for i in true_labels:
            true_labels_chinese.append(self.id2label[i])          
        for i in pred_labels:          
            pred_labels_chinese.append(self.id2label[i])

        #result = compute_metrics(preds_hanzi, out_label_hanzi)  # pred, true
        metric_dict = compute_metrics(true_labels, pred_labels, average='macro', decimal_place=4)  # pred, true

        metric_dict.update(
            {'prob_confidence': pred_positive_probs,
             'gt':true_labels})

        self.logger.info('\n'+classification_report(true_labels_chinese, pred_labels_chinese, digits=4))
        self.logger.info('\n'+f'灵敏度： {metric_dict["sensitivity"]}; 特异度： {metric_dict["specificity"]}')
        self.logger.info('\n'+str(confusion_matrix(true_labels_chinese, pred_labels_chinese)))
       
        with open (self.metric_file_path,'a') as fs:      
            print (classification_report(true_labels_chinese, pred_labels_chinese, digits=4), file = fs)
            print('\n')
            for metric_type in [ "specificity", "sensitivity", "weighted_f1_score", "acc", "benign_malig_acc", "kappa"]:
                print(f'{metric_type}: {metric_dict[metric_type]}', file = fs)
            print(confusion_matrix(true_labels_chinese, pred_labels_chinese), file = fs)
            print('\n', file = fs)

        if record_loss:
            return loss_meter.value()[0], metric_dict
        else:
            return metric_dict
        # return metric_dict, confidence
      

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params
