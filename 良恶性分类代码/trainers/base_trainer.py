import argparse
import datetime
import logging
import os
import time
import sys
from typing import List, Dict, Tuple
import shutil
import torch
from torch.nn import DataParallel
from torch.optim import Optimizer
import torchvision.models as models
import util
from util import init_logger


class BaseTrainer:
    """ Trainer base class with common methods """
    def __init__(self, args: argparse.Namespace):

        # meta_dataset: 元数据（甲状腺、淋巴结、乳腺等...）
        # dataset_name: 数据处理的方法（Resize、Keep-ratio等）
        args.running_name = f'{args.meta_dataset}_{args.dataset_name}_batch:{args.train_batch_size}_lr:{args.lr}_warmup:{args.lr_warmup}'

        if hasattr(args, 'hospital_name'):
            mark = args.CLS + '/' + args.hospital_name
        else:
            mark = args.CLS

        args.save_weight_path = mark  + '/' + args.running_name
        args.log_dir = f'logs/{mark}' + '/' + args.running_name
        self.args = args
        #* set seed
        if args.seed is not None:
            util.set_seed(args.seed)

         #* define a visualizer
        run_key = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))


        #* model save path
        if hasattr(args, 'model_save_path'):
            if hasattr(args, 'setting_name'):
                self.root_path = os.path.join(self.args.model_save_path, self.args.save_weight_path, self.args.setting_name)
            else:
                self.root_path = os.path.join(self.args.model_save_path, self.args.save_weight_path)
            util.create_directories_dir(self.root_path)
            self.save_path = os.path.join(self.root_path, self.args.model_name + f'_train_id:{self.args.train_id}',  run_key)
            util.create_directories_dir(self.save_path)

        #* init logger
        if hasattr(args, 'log_dir'):
            if args.running_mode == 'train':
                self.log_path = os.path.join(self.args.log_dir, 'train', self.args.model_name + '_' + f'train_id:{self.args.train_id}', run_key)
            else:
                self.log_path = os.path.join(self.args.log_dir, 'test', self.args.model_name,  run_key)
            util.create_directories_dir(self.log_path)
            init_logger(self.log_path)
            self.logger = logging.getLogger('main_log.base_trainer')
            if self.args.level=='debug':
                self.logger.setLevel(logging.DEBUG)
            elif self.args.level=='warning':
                self.logger.setLevel(logging.WARNING)
            elif self.args.level=='error':
                 self.logger.setLevel(logging.ERROR)
            else:
                 self.logger.setLevel(logging.INFO)
            self._save_arguments()
            if self.args.running_mode!='predict':
                self.metric_file_path=os.path.join(self.log_path, 'metric.txt')

        #* CUDA devices
        self.device = torch.device(f"cuda:{self.args.device_id}" if torch.cuda.is_available() and not args.cpu else "cpu")
        if torch.cuda.is_available() :
            torch.backends.cudnn.deterministic = True
        #self._gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        #* to record best model’s performence
        self.best_performace = None
        self.topK_performace2model = {}
        self.topK_recall_performace2model = {}
        self.eval_metric = args.metric_method

    def _save_arguments(self):
        """保存命令行参数至日志文件夹"""
        util.save_dict(self.log_path, self.args, 'args')

    def save_topK(self, num : int, model: models, optimizer: Optimizer, scheduler: None, performace: float, epoch: int, extra=None):
        """保存表现最好的K个模型"""
        if self.best_performace is None or performace > self.best_performace:
            self.best_performace = performace
            self.count_epoch = 0
        if self.best_performace is not None and performace <= self.best_performace:
            self.count_epoch += 1
        if self.count_epoch>=self.args.early_stop_epoch:
            self.logger.info("连续{}个epoch没有更高的performace,终止训练!!!".format(self.args.early_stop_epoch))
            sys.exit()
        model_save_dir = os.path.join(self.save_path,f'epo_{epoch}_{self.args.metric_method}_{performace}')
        if len(self.topK_performace2model) < num:
            self.topK_performace2model[str(epoch)+'_'+str(performace)] = model_save_dir
            self._save_model(model_save_dir,
                             model,
                             optimizer,
                             scheduler,
                             epoch,
                             extra=extra)
        else:
            cur_min_performace = min([float(i.split('_')[1]) for i in list(self.topK_performace2model.keys())])
            if performace > cur_min_performace:
                self.topK_performace2model[str(epoch)+'_'+str(performace)] = model_save_dir
                self._save_model(model_save_dir,
                                model,
                                optimizer,
                                scheduler,
                                epoch,
                                extra=extra)

                min_epo=100000
                for k in self.topK_performace2model:
                    temp_epo, temp_performace = int(k.split('_')[0]), float(k.split('_')[1])
                    if temp_performace == cur_min_performace:
                        min_epo = min(min_epo, temp_epo)
                min_key = str(min_epo)+'_'+str(cur_min_performace)
                shutil.rmtree(self.topK_performace2model[min_key])
                del self.topK_performace2model[min_key]

    def save_top_recall_model(self,
                              model: models,
                              optimizer: Optimizer,
                              scheduler: None,
                              malign_performance: float,
                              benign_performance: float,
                              epoch: int,
                              num:int,
                              extra=None):
        """保存表现最好的K个模型"""
        if (malign_performance > 0.7 and benign_performance > 0.7) or (malign_performance > 0.8 and benign_performance > 0.65):
            top_recall_save_path = os.path.join(self.save_path, 'top_recall_model')
            os.makedirs(top_recall_save_path, exist_ok=True)
            performance = round((malign_performance + benign_performance)/2, 4)
            model_save_dir = os.path.join(top_recall_save_path,
                                          f'epo_{epoch}_malign:{malign_performance}_belign:{benign_performance}')
            if len(self.topK_recall_performace2model) < num:
                self.topK_recall_performace2model[str(epoch) + '_' + str(performance)] = model_save_dir
                self._save_model(model_save_dir,
                                 model,
                                 optimizer,
                                 scheduler,
                                 epoch,
                                 extra=extra)
            else:
                cur_min_performace = min([float(i.split('_')[1]) for i in list(self.topK_recall_performace2model.keys())])
                if performance > cur_min_performace or malign_performance > 0.8 and benign_performance > 0.7:
                    self.topK_recall_performace2model[str(epoch) + '_' + str(performance)] = model_save_dir
                    self._save_model(model_save_dir,
                                     model,
                                     optimizer,
                                     scheduler,
                                     epoch,
                                     extra=extra)

                    min_epo = 100000
                    for k in self.topK_recall_performace2model:
                        temp_epo, temp_performace = int(k.split('_')[0]), float(k.split('_')[1])
                        if temp_performace == cur_min_performace:
                            min_epo = min(min_epo, temp_epo)
                    min_key = str(min_epo) + '_' + str(cur_min_performace)
                    shutil.rmtree(self.topK_recall_performace2model[min_key])
                    del self.topK_recall_performace2model[min_key]
    def _save_model(self, save_dir: str,
                    model: models,
                    optimizer: Optimizer = None,
                    scheduler=None,
                    epoch=None,
                    extra: dict = None, ):
        extra_state = {'epoch': epoch,
                       'optimizer': optimizer,
                       'scheduler': scheduler
                       }
        """保存模型参数"""
        if extra:
            extra_state.update(extra)

        util.create_directories_dir(save_dir)
        # * save model
        torch.save(model, os.path.join(save_dir, 'model.pt'))
        # * save args
        torch.save(self.args, os.path.join(save_dir, 'training_args.bin'))
        # * save extra
        state_path = os.path.join(save_dir, 'extra.state')
        torch.save(extra_state, state_path)
        def _save_model_rlb(self, state, epoch, save_mode):
            _file_path = os.path.join(self.save_path, '{}_{}'.format(epoch, save_mode))
            torch.save(state, _file_path)
            return 'The_{}_model have been saved in {}'.format(save_mode, _file_path)

    def _get_lr(self, optimizer):
        lrs = []
        for group in optimizer.param_groups:
            lr_scheduled = group['lr']
            lrs.append(lr_scheduled)
        return lrs

