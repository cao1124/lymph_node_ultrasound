import os
import sys
import cv2
import json
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
import logging
from dataset.augumentations import *
logger = logging.getLogger('main_log.dataset')

transformations = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ]
)

class BaseDatasets(Dataset):
    def __init__(self,
                 domain_dir,
                 domain_list,
                 boxes_path,
                 dataset_type,
                 patch_size,
                 crop_type,
                 is_rgb=None,
                 transforms=transformations,
                 aug_prob=None,
                 gain_aug_prob=None,
                 gain_range=None,
                 crop_ext_ratio_range=(-0.05, 0.2),
                 **kwargs):

        self.domain_dir = domain_dir
        self.domain_list = domain_list
        self.boxes_path = boxes_path
        self.dataset_type = dataset_type
        self.patch_size = patch_size
        self.is_rgb = is_rgb
        self.transforms = transforms
        self.augmentations = get_14_augu_compose(prob=0.2)
        self.aug_prob = aug_prob
        self.gain_aug_prob = gain_aug_prob
        self.gain_aug_prob = gain_aug_prob
        self.gain_range = gain_range
        self.crop_ext_ratio_range = crop_ext_ratio_range
        self.crop_type = crop_type
        self.kwargs = kwargs

    def __getitem__(self, item):
        raise NotImplementedError("This method must be implemented by the subclass")

    def __len__(self):
        raise NotImplementedError
    def _batch_extraction(self, image, img_info):
        raise NotImplementedError("This method must be implemented by the subclass")

    def _get_domain_data(self, **kwargs):
        raise NotImplementedError("This method must be implemented by the subclass")

    def _collect_fn(self, item):
        raise NotImplementedError("This method must be implemented by the subclass")

    def _random_decimal(self, start, end):
        random_num = random.uniform(start, end)
        rounded_num = round(random_num, 2)
        return rounded_num

    def _read_image(self, image_path:str,
                    is_rgb:bool=True, rgb2bgr:bool=True):

        if not is_rgb:
            return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        else:
            if not rgb2bgr:
                return cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:
                return cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    def _assign_label(self, tags):
        is_string = isinstance(tags, str)
        is_int = isinstance(tags, int)
        if is_string:
            signal_label = 1 if 'malign' == tags else 0
            return signal_label
        elif is_int:
            return tags
        else:
            if 'malign' in tags:
                return 1
            else:
                return 0
