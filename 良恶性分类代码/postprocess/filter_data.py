import os
import torch
import torch.nn as nn
import pickle
import cv2
import numpy as np
from tqdm import tqdm
from preprocess.about_crop import *
from torchvision.transforms import transforms

k_fold_path = '../refer/中山医院淋巴结malignNod_5_fold.pkl'
with open(k_fold_path, 'rb') as fs:
    k_fold_data = pickle.load(fs)


nod_cls_model_path_dict = \
    {
        'fold_1': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/中山医院-张伟-数据一致性/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:0/2024_11_27_16_08_51/top_recall_model/epo_11_malign:0.7306_belign:0.707/model.pt',
        'fold_2': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/中山医院-张伟-数据一致性/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_11_27_16_17_12/top_recall_model/epo_9_malign:0.7435_belign:0.7183/model.pt',
        'fold_3': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/中山医院-张伟-数据一致性/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:2/2024_11_27_16_26_24/top_recall_model/epo_12_malign:0.75_belign:0.7155/model.pt',
        'fold_4': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/中山医院-张伟-数据一致性/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:3/2024_11_27_16_35_27/epo_7_weighted_f1_score_0.726/model.pt',
        'fold_5': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/中山医院-张伟-数据一致性/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:4/2024_11_27_16_43_28/epo_6_weighted_f1_score_0.7189/model.pt',
    }

malign_cls_model_path_dict = \
    {
        'fold_1':'/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴瘤vs恶性-数据交叉验证/中山医院/malignNodClassification_KEEP_RATIO_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:0/2024_12_09_17_08_32/epo_97_weighted_f1_score_0.7779/model.pt',
        'fold_2':'/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴瘤vs恶性-数据交叉验证/中山医院/malignNodClassification_KEEP_RATIO_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:1/2024_12_09_17_20_11/epo_70_weighted_f1_score_0.797/model.pt',
        'fold_3':'/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴瘤vs恶性-数据交叉验证/中山医院/malignNodClassification_KEEP_RATIO_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:2/2024_12_09_17_31_41/epo_67_weighted_f1_score_0.7677/model.pt',
        'fold_4':'/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴瘤vs恶性-数据交叉验证/中山医院/malignNodClassification_KEEP_RATIO_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:3/2024_12_09_17_42_12/epo_97_weighted_f1_score_0.7839/model.pt',
        'fold_5': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴瘤vs恶性-数据交叉验证/中山医院/malignNodClassification_KEEP_RATIO_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:4/2024_12_09_17_52_14/epo_11_weighted_f1_score_0.7646/model.pt'
    }


def image2patch(image, box):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    patch, roi = crop_roi_extend(box, image, ratio=0.25, size=image.shape[:2])

    resized_patch = resize_img_keep_ratio(patch, target_size=(224, 224))
    return resized_patch

def transform_patch(patch, device='cuda:0'):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans  = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )
    patch_tensor = trans(patch)
    return patch_tensor.unsqueeze(0).to(device)





def collect_patch_label(domain_dirs, loaders):

    patches = []
    true_labels = []
    image_names = []

    for sample in tqdm(loaders, total=len(loaders), desc='collecting patch label'):
        box = sample['boxes']
        image_name = sample['image_name']
        image_path = os.path.join(domain_dirs, image_name)
        int_box = list(map(int, box.split(',')))
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        patch = image2patch(image, int_box)
        patch_tensor = transform_patch(patch)
        patches.append(patch_tensor)
        image_names.append(image_name)
        tags = sample['tags']
        if 'lymphomas' in tags:
            true_labels.append(0)
        else:
            true_labels.append(1)

    return torch.concat(patches, dim=0), true_labels, image_names


def main(domain_dirs,
         all_loaders,
         fold_num,
         device='cuda:0'):

    loaders = all_loaders[fold_num]['test']
    cls_model = torch.load(malign_cls_model_path_dict[fold_num], map_location='cpu')
    cls_model.to(device)
    patch_tensor, true_labels, image_names = collect_patch_label(domain_dirs, loaders)
    with torch.no_grad():
        outputs = cls_model(patch_tensor)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()

    filter_image_names = [image_names[i] for i in range(len(image_names)) if preds[i] == true_labels[i]]
    return filter_image_names






if __name__ == '__main__':

    domain_dir = '/mnt/data/hsy/数据/淋巴结分类数据/中山医院'
    save_dirs = '/mnt/data/hsy/科研项目/良恶性分类代码/filter_data/上海中山恶性亚分分型'
    all_loaders = k_fold_data
    for fold_num in ['fold_1', 'fold_2', 'fold_3', 'fold_4', 'fold_5']:
        filter_image_names = main(domain_dir, all_loaders, fold_num)

        for filter_image_name in tqdm(filter_image_names, desc='saving filter image', unit='image'):
            _split = filter_image_name.rsplit('/', 3)
            hosp_name, pat_name, image_name = _split[0], _split[1], _split[-1]
            hosp_pat_image_save_path = os.path.join(save_dirs, hosp_name, pat_name, 'images')
            hosp_annotation_save_path = os.path.join(save_dirs, hosp_name, pat_name, 'annotations')

            os.makedirs(hosp_pat_image_save_path, exist_ok=True)
            os.makedirs(hosp_annotation_save_path, exist_ok=True)

            or_image_path = os.path.join(domain_dir, filter_image_name)
            or_annotation_path = or_image_path.replace('images', 'annotations') + '.json'


            shutil.copy(or_image_path, os.path.join(hosp_pat_image_save_path, image_name))
            shutil.copy(or_annotation_path, os.path.join(hosp_annotation_save_path, image_name + '.json'))

