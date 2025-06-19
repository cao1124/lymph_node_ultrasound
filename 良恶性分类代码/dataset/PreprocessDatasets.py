import os
import sys
import cv2
import json
import albumentations as A
import random
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset
from torchvision.transforms import transforms as TR
from PIL import Image
import logging
from util import binary_search_first_ge, find_num
from dataset.augumentations import get_gain_aug, get_14_augu_compose, get_zhanglei_aug, get_transformations
from preprocess.about_crop import crop_roi_extend, resize_img_keep_ratio
from dataset._basedatasets import transformations, BaseDatasets, logger
from dataset._basetool import BoxTuneTool
from typing import Union, List, Dict, Tuple

class KeepRatioResizeDatasets(BaseDatasets):
    # 随机外扩 + 等比例缩放
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
        super().__init__(
            domain_dir,
            domain_list,
            boxes_path,
            dataset_type,
            patch_size,
            crop_type,
            is_rgb,
            transforms,
            aug_prob,
            gain_aug_prob,
            gain_range,
            crop_ext_ratio_range,
            **kwargs
        )
        if kwargs.get('OutDomainTest') or kwargs.get('ZhongShan'):
            self.patch_list = self._get_out_domain_data(dataset_type, boxes_path, domain_list)
        elif kwargs.get('K_Fold'):
            self.patch_list = self._get_k_fold_data(dataset_type, boxes_path, domain_list, kwargs.get('fold_num'))
        elif kwargs.get('HX'):
            self.patch_list = self._get_hx_data(dataset_type, boxes_path)
        else:
            self.patch_list = self._get_domain_data(dataset_type, boxes_path, domain_list)


        self.length = len(self.patch_list)
    def _get_domain_data(self,
                        dataset_type:str,
                        boxes_path:str,
                        domain_list:Union[List, None]):
        all_img_list = []
        assert boxes_path.split('.')[-1] in ['pkl', 'pickle'], f"boxes_path {boxes_path} should be a pickle file"
        with open(boxes_path, 'rb') as fs:
            all_boxes_info = pickle.load(fs)
            domain_data_list = all_boxes_info[dataset_type]
            logger.info(f'类型：{dataset_type}，数据量：{len(domain_data_list)}')
            all_img_list.extend(domain_data_list)
        logger.info(f'{dataset_type}_总数据量：{len(all_img_list)}')
        return all_img_list
    def _get_out_domain_data(self,
                             dataset_type:str,
                             boxes_path:str,
                             domain_list:Union[List, None]):
        # outDomain Data 全部用作测试集, pkl文件不再以dataset_type进行区分, 而是以不同的医院进行区分
        all_img_list = []
        assert boxes_path.split('.')[-1] in ['pkl', 'pickle'], f"boxes_path {boxes_path} should be a pickle file"
        with open(boxes_path, 'rb') as fs:
            all_boxes_info = pickle.load(fs)
            for domain in all_boxes_info:
                if domain_list is not None and domain not in domain_list:
                    continue
                else:
                    hosp_domain_data = all_boxes_info[domain][dataset_type]
                    all_img_list.extend(hosp_domain_data)
                    logger.info(f'{domain}_{dataset_type}_总数据量：{len(hosp_domain_data)}')
        return all_img_list

    def _get_k_fold_data(self,
                         dataset_type:str,
                         boxes_path:str,
                         domain_list:Union[List, None],
                         fold_num:str,):

        all_img_list = []
        assert boxes_path.split('.')[-1] in ['pkl', 'pickle'], f"boxes_path {boxes_path} should be a pickle file"
        with open(boxes_path, 'rb') as fs:
            all_boxes_info = pickle.load(fs)
        k_fold_data = all_boxes_info[fold_num][dataset_type]
        all_img_list.extend(k_fold_data)
        logger.info(f'{fold_num}_{dataset_type}_总数据量：{len(all_img_list)}')
        return all_img_list


    def _get_hx_data(self,
                     dataset_type,
                     boxes_path):

        all_img_list = []
        assert boxes_path.split('.')[-1] in ['pkl', 'pickle'], f"boxes_path {boxes_path} should be a pickle file"
        with open(boxes_path, 'rb') as fs:
            all_boxes_info = pickle.load(fs)
            for domain in all_boxes_info:
                hosp_domain_data = all_boxes_info[domain][dataset_type]
                all_img_list.extend(hosp_domain_data)
                logger.info(f'{domain}_{dataset_type}_总数据量：{len(hosp_domain_data)}')
        return all_img_list

    def _collect_fn(self, item):

        img_info = self.patch_list[item]
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.domain_dir, img_info['image_name'].replace('.jpg', '.png'))
        image = self._read_image(image_path, self.is_rgb)
        box =  list(map(int, img_info['boxes'].split(',')))
        if self.dataset_type == 'train':
            crop_ext_ratio = self._random_decimal(self.crop_ext_ratio_range[0], self.crop_ext_ratio_range[1])
        else:
            crop_ext_ratio = 0.2
        img_h, img_w = image.shape[:2]
        img_size = [img_h, img_w]
        det_img, roi = crop_roi_extend(box, image, crop_ext_ratio, img_size)
        patch = resize_img_keep_ratio(det_img)
        return patch, img_info

    def __getitem__(self, item):
        patch, img_info = self._collect_fn(item)
        if self.dataset_type == 'train':
            rd = random.random()
            if rd <= self.gain_aug_prob:
                patch = get_gain_aug(image=patch, gain_range=self.gain_range)
            elif rd > self.gain_aug_prob and rd <= self.gain_aug_prob + self.aug_prob:
                sample = self.augmentations(image=patch)
                patch = sample['image']

        if self.transforms is not None:
            patch_tensor = self.transforms(patch)
        else:
            patch_tensor = patch / 255.0
            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).float()

        label = self._assign_label(img_info['tags'])
        return (torch.FloatTensor(patch_tensor), label, patch,os.path.join(self.domain_dir, img_info['image_name']))

    def __len__(self):
        return self.length

class ResizeDatasets(KeepRatioResizeDatasets):
    # 固定外扩 + 线性插值
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
        super().__init__(
            domain_dir,
            domain_list,
            boxes_path,
            dataset_type,
            patch_size,
            crop_type,
            is_rgb,
            transforms,
            aug_prob,
            gain_aug_prob,
            gain_range,
            crop_ext_ratio_range,
            **kwargs
        )
    def _collect_fn(self, item):

        img_info = self.patch_list[item]
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.domain_dir, img_info['image_name'].replace('.jpg', '.png'))
        image = self._read_image(image_path, self.is_rgb)
        box = list(map(int, img_info['boxes'].split(',')))
        crop_ext_ratio = 0
        img_h, img_w = image.shape[:2]
        img_size = [img_h, img_w]
        det_img, roi = crop_roi_extend(box, image, crop_ext_ratio, img_size)
        patch = cv2.resize(det_img, (self.patch_size, self.patch_size))
        return patch, img_info


    def __getitem__(self, item):
        return super().__getitem__(item)

    def __len__(self):
        return self.length

class WindowCropCenterDatasets(KeepRatioResizeDatasets):
    # 中心Window裁剪 + 隐式提示
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
        super().__init__(
            domain_dir,
            domain_list,
            boxes_path,
            dataset_type,
            patch_size,
            crop_type,
            is_rgb,
            transforms,
            aug_prob,
            gain_aug_prob,
            gain_range,
            crop_ext_ratio_range,
            **kwargs
        )
        self.kwargs = kwargs
        self.pool_size = [256, 448, 512, 600, 700, 800, 900, 1024]
    def _collect_fn(self, item):

        img_info = self.patch_list[item]
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.domain_dir, img_info['image_name'].replace('.jpg', '.png'))
        image = self._read_image(image_path, self.is_rgb)
        image_height, image_width = image.shape[:2]
        x, y, w, h = list(map(int, img_info['boxes'].split(',')))

        window_size = self.binary_search_first_ge(self.pool_size, max(w, h))
        CropTool = BoxTuneTool(
            box=[x, y, w, h],
            img_size=(image_height, image_width)
        )
        new_x, new_y, new_w, new_h = CropTool.get_center_box(window_size=(window_size, window_size))
        patch, roi = crop_roi_extend([new_x, new_y, new_w, new_h], image, 0, [image_height, image_width])
        img_info.update(
            {
                'scaled_boxes': [
                    x - new_x,
                    y - new_y,
                    w,
                    h
                ]
            }
        )
        return patch, img_info

    def __getitem__(self, item):

        patch, img_info = self._collect_fn(item)
        x, y, w, h = list(map(int, img_info['scaled_boxes']))
        old_shape = patch.shape[:2]
        if patch.shape[:2] != (self.patch_size, self.patch_size):
            patch = cv2.resize(patch, (self.patch_size, self.patch_size))
            new_shape = patch.shape[:2]
            x_left = int(x * new_shape[1] / old_shape[1])
            x_right = int((x + w) * new_shape[1] / old_shape[1])
            y_top = int(y * new_shape[0] / old_shape[0])
            y_bottom = int((y + h) * new_shape[0] / old_shape[0])
        else:
            x_left, x_right, y_top, y_bottom = x, x + w, y, y + h
            patch = patch
        cv2.rectangle(patch, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 1)
        image_path = os.path.join(self.domain_dir, img_info['image_name'])

        # 数据增强
        if self.dataset_type == 'train':
            rd = random.random()
            if rd <= self.gain_aug_prob:
                patch = get_gain_aug(image=patch, gain_range=self.gain_range)
            elif rd > self.gain_aug_prob and rd <= self.gain_aug_prob + self.aug_prob:
                sample = self.augmentations(image=patch)
                patch = sample['image']
        if self.transforms is not None:
            patch_tensor = self.transforms(patch)
        else:
            patch = patch / 255.0
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()
        # 信息返回
        signal_label = self._assign_label(img_info['tags'])
        return (torch.FloatTensor(patch_tensor), signal_label, patch, image_path)

    def __len__(self):
        return self.length
    def binary_search_first_ge(self, arr:List[int], target:int):
        left, right = 0, len(arr) - 1
        result = -1
        while left <= right:
            mid = left + (right - left) // 2

            if arr[mid] >= target:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        return arr[result]

class WindowCropCenterExplictPrompt(WindowCropCenterDatasets):
    # 中心Crop裁剪 + 显式提示

    def __init__(self,
                 domain_dir,
                 domain_list,
                 boxes_path,
                 dataset_type,
                 patch_size,
                 crop_type,
                 is_rgb=None,
                 transforms=None,
                 aug_prob=None,
                 gain_aug_prob=None,
                 gain_range=None,
                 crop_ext_ratio_range=(-0.05, 0.2),
                 **kwargs):
        super(WindowCropCenterDatasets, self).__init__(
            domain_dir,
            domain_list,
            boxes_path,
            dataset_type,
            patch_size,
            crop_type,
            is_rgb,
            transforms,
            aug_prob,
            gain_aug_prob,
            gain_range,
            crop_ext_ratio_range,
            **kwargs
        )
        self.pool_size = [224, 256, 448, 512, 600, 700, 800, 900, 1024]
    def _collect_fn(self, item):
        img_info = self.patch_list[item]
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.domain_dir, img_info['image_name'].replace('.jpg', '.png'))
        image = self._read_image(image_path, is_rgb=False)

        mask_path = image_path.replace('images', 'masks')
        if not os.path.exists(mask_path):
            mask_path = mask_path.replace('.jpg', '.png')
            if not os.path.exists(mask_path):
                try:
                    contour_array = find_num(img_info['contour'])
                    _mask = np.zeros_like(image)
                    cv2.drawContours(
                        _mask,
                        [contour_array],
                        -1,
                        (255, 255, 255),
                        -1
                    )
                except:
                    raise ValueError('please check the image integrity and the contour info')
            else:
                _mask = self._read_image(mask_path, is_rgb=False)

        image_height, image_width = image.shape[:2]
        x, y, w, h = list(map(int, img_info['boxes'].split(',')))

        mask = np.zeros_like(_mask)
        mask[y:y+h, x:x+w] = _mask[y:y+h, x:x+w]

        window_size = self.binary_search_first_ge(self.pool_size, max(w, h))
        CropTool = BoxTuneTool(
            box=[x, y, w, h],
            img_size=(image_height, image_width)
        )
        # scale_factor = 100
        new_x, new_y, new_w, new_h = CropTool.get_center_box(window_size=(window_size, window_size))
        patch, roi = crop_roi_extend([new_x, new_y, new_w, new_h], image, 0, [image_height, image_width])
        mask_patch, _ = crop_roi_extend([new_x, new_y, new_w, new_h], mask, 0, [image_height, image_width])
        img_info.update(
            {
                'scaled_boxes': [
                    x - new_x,
                    y - new_y,
                    w,
                    h
                ],
                'mask_patch': mask_patch,
                'image_patch': patch,
                'roi_region': roi,
                'pixel_area_ratio': int(255 * w * h / (image_height * image_width))
            }
        )
        return patch, img_info

    def __getitem__(self, item):

        image_patch, img_info = self._collect_fn(item)
        mask_patch = img_info['mask_patch']
        pixel_area_ratio = img_info['pixel_area_ratio']
        assert image_patch.shape[:2] == mask_patch.shape[:2], f"patch shape {image_patch.shape} != mask shape {mask_patch.shape}"
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if image_patch.shape[:2] != (self.patch_size, self.patch_size):
            image_patch = cv2.resize(image_patch, (self.patch_size, self.patch_size))
        if mask_patch.shape[:2] != (self.patch_size, self.patch_size):
            mask_patch = cv2.resize(mask_patch, (self.patch_size, self.patch_size))
        pixel_area_ratio_change = np.ones_like(mask_patch) * pixel_area_ratio

        # 数据增强
        # if self.dataset_type == 'train':
        #     image_patch = np.concatenate([image_patch[..., np.newaxis], mask_patch[..., np.newaxis], pixel_area_ratio_change[..., np.newaxis]], axis=-1)
        #     rd = random.random()
        #     if rd <= self.gain_aug_prob:
        #         image_patch = get_gain_aug(image=image_patch, gain_range=self.gain_range)
        #     elif rd > self.gain_aug_prob and rd <= self.gain_aug_prob + self.aug_prob:
        #         sample = self.augmentations(image=image_patch)
        #         image_patch = sample['image']
        #     image_patch = np.mean(image_patch, axis=-1, keepdims=False)

        patch = np.concatenate(
            [image_patch[..., np.newaxis], mask_patch[..., np.newaxis], pixel_area_ratio_change[..., np.newaxis]],axis=-1)

        if self.transforms is not None:
            patch_tensor = self.transforms(patch)
        else:
            patch_tensor = patch.astype(np.float32)
            patch_tensor /= 255.0
            patch_tensor = torch.from_numpy(patch_tensor).permute(2, 0, 1).float()
        # 信息返回
        signal_label = self._assign_label(img_info['tags'])
        return (torch.FloatTensor(patch_tensor), signal_label, patch, image_path)



class NoShuffleKeepData(KeepRatioResizeDatasets):
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
        super().__init__(
            domain_dir,
            domain_list,
            boxes_path,
            dataset_type,
            patch_size,
            crop_type,
            is_rgb,
            transforms,
            aug_prob,
            gain_aug_prob,
            gain_range,
            crop_ext_ratio_range,
            **kwargs
        )

        # 保证一个patch里的样本都是相同类别的样本
        if self.dataset_type == 'val' or self.dataset_type == 'test':
            self.patch_list = self.keep_no_shuffle()
            self.length = len(self.patch_list)

        self.transform_train = TR.Compose(
            [
                TR.RandomResizedCrop((224, 224), scale=(0.05, 1.0)),
                TR.ToTensor(),
                TR.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )

        self.transform_val = TR.Compose(
            [
                TR.Resize((224, 224)),
                TR.ToTensor(),
                TR.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ]
        )


    def keep_no_shuffle(self):
        batch_size = 32
        malign_list = []
        benign_list = []
        for patch in self.patch_list:
            if self._assign_label(patch['tags']) == 1:
                malign_list.append(patch)
            elif self._assign_label(patch['tags']) == 0:
                benign_list.append(patch)
        malign_num = (len(malign_list) // batch_size) * batch_size
        benign_num = (len(benign_list) // batch_size) * batch_size
        return malign_list[:malign_num] + benign_list[:benign_num]

    def _collect_fn(self, item):

        img_info = self.patch_list[item]
        image_path = os.path.join(self.domain_dir, img_info['image_name'])
        if not os.path.exists(image_path):
            image_path = os.path.join(self.domain_dir, img_info['image_name'].replace('.jpg', '.png'))
        image = self._read_image(image_path, self.is_rgb)
        box = list(map(int, img_info['boxes'].split(',')))
        if self.dataset_type == 'train':
            crop_ext_ratio = self._random_decimal(self.crop_ext_ratio_range[0], self.crop_ext_ratio_range[1])
        else:
            crop_ext_ratio = 0.25
        img_h, img_w = image.shape[:2]
        img_size = [img_h, img_w]
        det_img, roi = crop_roi_extend(box, image, crop_ext_ratio, img_size)
        patch = resize_img_keep_ratio(det_img)
        return patch, img_info

    def __getitem__(self, item):
        patch, img_info = self._collect_fn(item)
        if self.dataset_type == 'train':
            rd = random.random()
            if rd <= self.gain_aug_prob:
                patch = get_gain_aug(image=patch, gain_range=self.gain_range)
            elif rd > self.gain_aug_prob and rd <= self.gain_aug_prob + self.aug_prob:
                sample = self.augmentations(image=patch)
                patch = sample['image']
        if self.transforms is not None:
            patch_tensor = self.transforms(patch)
        else:
            patch = patch / 255.0
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float()

        # patch = Image.fromarray(patch)
        # if self.dataset_type == 'train':
        #     patch_tensor = self.transform_train(patch)
        # else:
        #     patch_tensor = self.transform_val(patch)

        label = self._assign_label(img_info['tags'])
        return (torch.FloatTensor(patch_tensor), label, np.array(patch), os.path.join(self.domain_dir, img_info['image_name']))

