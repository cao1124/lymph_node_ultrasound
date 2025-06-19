import argparse
import os

from config import getconfig

task = 'lymph_nod_cls'
def common_args(arg_parser):

    arg_parser.add_argument('--config', type=str)
    arg_parser.add_argument('--cpu', action='store_true', default=False, help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add_argument('--seed', type=int, default=1000, help="Seed")
    arg_parser.add_argument('--device_id', type=str, help="gpu device id to use",default='1')
    arg_parser.add_argument('--ngpu', type=int, help='use multi gpu to train', default=1)
    arg_parser.add_argument('--device_ids', type=str, default='0')
    arg_parser.add_argument('--target_size', type=int, default=224, help="target size")
    arg_parser.add_argument('--level',type=str, default='info', help="logging level:debug/info/error/warning")
    arg_parser.add_argument('--metric_method', type=str, default="weighted_f1_score", help="Metric method", choices=["weighted_f1_score", "macro_f1_score", "acc", "benign_malig_acc"])
    arg_parser.add_argument('--project', type=str, default='ZhongShan_malign_cls', help='Name of project for wandb')
    arg_parser.add_argument('--model_path', type=str, default='', help='the weight path of test model')
    arg_parser.add_argument('--dataset_name', '-dn', type=str, default='RESIZE', choices=['RESIZE', 'KEEP_RATIO', 'WINDOW_CROP_CENTER','WINDOW_CROP_CENTER_EXPLICIT_PROMPT','NoShuffleKeepRatio'])
    arg_parser.add_argument('--crop_type', default='normal', type=str)

    return arg_parser

def train_argparser():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--mode', type=str, default='train', help='the train mode of model')
    arg_parser.add_argument('--train_id', type=int, default=0, help='the train times')

    # * Augmentation
    arg_parser.add_argument('--crop_ext_ratio_range', type=tuple, default=(0.05, 0.25), help="crop_ext_ratio_range")
    arg_parser.add_argument('--mixup', action='store_true', default=False)
    arg_parser.add_argument('--use_aug', action='store_true', default=True)
    arg_parser.add_argument('--is_rgb', action='store_true', default=True)
    arg_parser.add_argument('--gain_aug_prob', type=float, default=0.3, help="prob to use gain aug")
    arg_parser.add_argument('--aug_prob', type=float, default=0.5, help="prob to use else aug")
    arg_parser.add_argument('--gain_range', type=tuple, default=(25, 60), help="gain range")

    # * Logging
    arg_parser.add_argument('--eval_trainset', action='store_true', default=False,
                            help="If true:evaluate on train dataset before evaluate on validation dataset")
    arg_parser.add_argument('--use_tensorboard', action='store_true', default=False,
                            help="Use tensorboard to display lr and f1-score etc.")

    # * Model & Datasets
    arg_parser.add_argument('--model_name', type=str, default='efficientnet-b0', help="Name of model")
    arg_parser.add_argument('--model_save_path', type=str, help= "Path to directory where model checkpoints are stored",default='saved_weights/')
    arg_parser.add_argument('--optimizer_name', type=str, default="Adam", choices=["AdamW", "Adam", "SGD"], help="Name of optimizer")
    arg_parser.add_argument('--criterion_name', type=str, default='CrossEntropy', choices=['HardSampleLoss', "CrossEntropy", "LabelSmoothingCrossEntropy", "Weighted_CrossEntropy", 'FocalLoss','Weighted_Focal'],
                            help="Name of optimizer")
    arg_parser.add_argument('--fold_num', default='fold_1', type=str, help='K折交叉验证')

    # * lr
    arg_parser.add_argument('--use_weight_decay', action='store_true', default=False)
    arg_parser.add_argument('--weight_decay', type=float, default=0.0005, help="Weight decay to apply")
    arg_parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")
    arg_parser.add_argument("--dropout_rate", default=0.25, type=float, help="Dropout for fully-connected layers")
    arg_parser.add_argument('--save_model_num', type=int, default=3, help="Save top K best acc model")

    opt = getconfig(task)
    for key, value in opt.__dict__.items():
        if key != '__module__':
            arg_parser.add_argument(f'--{key}', default=value, type=type(value))

    common_args(arg_parser)
    return arg_parser

def test_argparser():
    arg_parser = argparse.ArgumentParser()
    #* Input
    arg_parser.add_argument('--dataset_type', type=str, default='test')
    arg_parser.add_argument('--model_name', type=str, default='eb0', help="Name of model")
    arg_parser.add_argument('--domain_list', type=str, default='refer/domain_list.txt', help="Path to domain list")
    arg_parser.add_argument('--out_domain_list', type=str, default='', help='Path to out domain list')
    arg_parser.add_argument('--is_rgb', action='store_true', default=True)
    opt = getconfig(task)
    for k, v in opt.__dict__.items():
        if not k.startswith('__'):
            arg_parser.add_argument(f'--{k}', default=v, type=type(v))
    common_args(arg_parser)
    return arg_parser


