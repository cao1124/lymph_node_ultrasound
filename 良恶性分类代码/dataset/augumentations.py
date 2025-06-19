import albumentations as albu
from albumentations import *
import cv2
import random
import numpy as np
from torchvision import transforms


def get_classical_aug():
    train_transform = albu.Compose(
        [
            albu.GaussNoise(p=0.2),
            albu.Resize(224, 224),
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(0.2),
            albu.Flip(p=0.2)
        ]
    )
    return train_transform


def get_zhanglei_aug(prob):
    train_transform = [
        albu.HorizontalFlip(p=prob),
        albu.VerticalFlip(p=prob),

        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.IAAPerspective(p=prob),

        # albu.RandomResizedCrop(224,224,p=0.8),
        # albu.RandomScale(p=1),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightnessContrast(p=1),
                # albu.RandomGamma(p=1, gamma_limit=(40,160))
                albu.RandomGamma(p=1)

             ], p=prob
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ], p=prob,
        ),
    ]

    return albu.Compose(train_transform)

def get_13_augu_compose(prob):
    train_transform = [
        RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=None, always_apply=False, p=prob),
        albu.OneOf(
            [   Rotate(p=prob, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(always_apply=False, p=prob),
                VerticalFlip(always_apply=False, p=prob),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=(1,1), rotate_limit=(0,0), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=prob)
            ],p=1
         ),
        albu.OneOf(
            [  
                GridDistortion(p=prob, distort_limit=(-0.2, 0), num_steps=5),
                ElasticTransform(p=prob, alpha_affine=30),
                Perspective(scale=(0.01,0.05), p=prob)
             ], p=1
        ),
         albu.OneOf(
            [
                GaussNoise(p=prob, var_limit=(20,100)),
                MotionBlur(blur_limit=15, always_apply=False, p=prob),
                GaussianBlur(blur_limit=15, always_apply=False, p=prob),
                RandomGamma(gamma_limit=(40, 160), eps=1e-07, always_apply=False, p=prob),
                CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob),
             ], p=1
        ),
    ]
    return albu.Compose(train_transform)

def get_14_augu_compose(prob):
    train_transform = [
        albu.OneOf(
            [   Rotate(p=prob, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(always_apply=False, p=prob),
                VerticalFlip(always_apply=False, p=prob),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=(1,1), rotate_limit=(0,0), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=prob)
            ],p=prob
         ),
        albu.OneOf(
            [  
                GridDistortion(p=prob, distort_limit=(-0.2, 0), num_steps=5),
                ElasticTransform(p=prob, alpha_affine=30),
                Perspective(scale=(0.01,0.05), p=prob)
             ], p=prob
        ),
         albu.OneOf(
            [
                RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=None, always_apply=False, p=prob),
                GaussNoise(p=prob, var_limit=(20,100)),
                MotionBlur(blur_limit=15, always_apply=False, p=prob),
                GaussianBlur(blur_limit=15, always_apply=False, p=prob),
                RandomGamma(gamma_limit=(40, 160), eps=1e-07, always_apply=False, p=prob),
                CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob),
                RGBShift(always_apply=False, p=prob, r_shift_limit=(-20, 20), g_shift_limit=(-20, 20), b_shift_limit=(-20, 20)),
             ], p=prob
        ),
    ]
    return albu.Compose(train_transform)

def get_13_augu(prob):
    train_transform = [
        albu.OneOf(
            [
                Rotate(p=1, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                GridDistortion(p=1, distort_limit=(-0.2, 0), num_steps=5),
                ElasticTransform(p=1, alpha_affine=30),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=(1,1), rotate_limit=(0,0), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=1),
                GaussNoise(p=1, var_limit=(20,100)),
                MotionBlur(blur_limit=15, always_apply=False, p=1),
                GaussianBlur(blur_limit=15, always_apply=False, p=1),
                HorizontalFlip(always_apply=False, p=1),
                VerticalFlip(always_apply=False, p=1),
                RandomGamma(gamma_limit=(40, 160), eps=1e-07, always_apply=False, p=1),
                RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=None, always_apply=False, p=1),
                CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=1),
                Perspective(scale=(0.01,0.05), p=1),
             ], p=prob
        ),
    ]
    return albu.Compose(train_transform)



def get_deformation_augu():
    prob = 1
    train_transform = [
        albu.OneOf(
            [  
                GridDistortion(p=prob, distort_limit=(-0.2, 0), num_steps=5),
                ElasticTransform(p=prob, alpha_affine=30),
                Perspective(scale=(0.01,0.05), p=prob)
             ], p=1
        )
    ]
    return albu.Compose(train_transform)

def get_position_augu():
    prob = 1
    train_transform = [
        albu.OneOf(
            [  
                Rotate(p=prob, limit=180, border_mode=cv2.BORDER_CONSTANT, value=0),
                HorizontalFlip(always_apply=False, p=prob),
                VerticalFlip(always_apply=False, p=prob),
                ShiftScaleRotate(shift_limit=0.0625, scale_limit=(1,1), rotate_limit=(0,0), interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=prob)
             ], p=1
        )
    ]
    return albu.Compose(train_transform)

def get_noise_augu():
    prob = 1
    train_transform = [
        albu.OneOf(
            [   RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, brightness_by_max=None, always_apply=False, p=prob),
                GaussNoise(p=prob, var_limit=(20,100)),
                MotionBlur(blur_limit=15, always_apply=False, p=prob),
                GaussianBlur(blur_limit=15, always_apply=False, p=prob),
                RandomGamma(gamma_limit=(40, 160), eps=1e-07, always_apply=False, p=prob),
                CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=prob)
             ], p=1
        )
    ]
    return albu.Compose(train_transform)


#* 超声机反向工程-模拟增益变化
def gain_pixel_transform(pixel, config):
        # 在这里定义您的像素变化函数
        transformed_pixel = pixel + 4.3*config# 根据像素值计算变化后的像素值
        if transformed_pixel<0:
            transformed_pixel=0
        if transformed_pixel>255:
            transformed_pixel=255
        return transformed_pixel

def gain_pixel_transform_rgb(pixel, config):
    # 在这里定义您的像素变化函数
    transformed_pixel = np.zeros_like(pixel)  # 创建与输入像素相同大小的空数组

    for i in range(len(pixel)):
        transformed_pixel[i] = pixel[i] + 4.3 * config  # 根据像素值计算变化后的像素值
        if transformed_pixel[i] < 0:
            transformed_pixel[i] = 0
        if transformed_pixel[i] > 255:
            transformed_pixel[i] = 255

    return transformed_pixel

def stimulate_gain(image, gain_range):
    # 将图像转换为灰度图像
    # gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # 定义函数来对像素值进行变化
   
    gain = random.choice(range(gain_range[0], gain_range[1]))
    # 定义查找表（Lookup Table）
    table = np.arange(256, dtype=np.uint8)  # 创建一个包含0到255的数组
    # 使用像素变换函数更新查找表
    for i in range(256):
        table[i] = gain_pixel_transform(i, gain-43)
    # 应用查找表，实现像素变换
    transformed_image = cv2.LUT(image, table)
    # transformed_image = cv2.cvtColor(transformed_image, cv2.COLOR_RGB2BGR)

    # cv2.imwrite('temp_aug.jpg',transformed_image)
    return transformed_image

def get_gain_aug(image, gain_range):
    return stimulate_gain(image, gain_range)


def get_transformations(type:str):
    if type == 'Imagenet':
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        transformations = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    elif type == 'nodule_data':
        mean = [0.07873619767751619, 0.08275937366200807, 0.08670211072693049]
        std = [0.03643061679767745, 0.03745543767326019, 0.03834949950641613]
        transformations = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    elif type == 'all_lymphatic_data':
        mean = [0.260, 0.260, 0.265]
        std = [0.044, 0.004, 0.045]
        transformations = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    elif type == 'gold_lymphatic_data':
        mean = [0.260, 0.260, 0.266]
        std = [0.048, 0.048, 0.050]
        transformations = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(mean, std)])
    elif type == 'mae_pretrain':
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        transformations = transforms.Compose([
            transforms.RandomResizedCrop(size=(224,224),scale=(0.2, 1.0)),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    else:
        raise TypeError('not supported type')

    return transformations

   
   
