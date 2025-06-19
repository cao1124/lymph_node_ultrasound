import os
import cv2
import json
import random
import pickle
from util import find_num, json2mm
from collections import defaultdict
from typing import List
from tqdm import tqdm

def split_K_fold(fold_num:int,
                 dirs:str,
                 hosp:str):

    hosp_dirs = os.path.join(dirs, hosp)
    patient_list = os.listdir(hosp_dirs)
    random.shuffle(patient_list)
    all_num = len(patient_list)
    n = all_num // fold_num
    remainder = all_num % fold_num
    start = 0
    k_folds_list = []
    end = 0
    for i in range(fold_num):
        end += n
        if remainder > 0:
            end += 1
            remainder -= 1
        k_folds_list.append(patient_list[start:end])
        start = end
    return k_folds_list

def generate_pkl(dirs:str,
                hosp:str,
                k_folds_list:List):
    # 根据划分的k折数据集，生成对应的pkl文件

    fold_num = len(k_folds_list)
    fold_pkl = defaultdict(dict)
    for i in range(fold_num):
        train_val_pats = []
        test_pats = k_folds_list[i]
        for j in range(fold_num):
            if i == j:
                continue
            train_val_pats.extend(k_folds_list[j])
        train_pats = train_val_pats[:int(len(train_val_pats)*0.8)]
        val_pats = train_val_pats[int(len(train_val_pats)*0.8):]
        train_test_val = {
            'train': collect_info(dirs, hosp, train_pats),
            'val': collect_info(dirs, hosp, val_pats),
            'test': collect_info(dirs, hosp, test_pats)
        }
        fold_pkl[f'fold_{i+1}'] = train_test_val

    return fold_pkl
def collect_info(dirs:str,
                 hosp:str,
                 pats:list):

    infos = []
    hosp_dirs = os.path.join(dirs, hosp)
    for pat in pats:
        pat_image_path = os.path.join(hosp_dirs, pat, 'images')
        pat_annotations_path = os.path.join(hosp_dirs, pat, 'annotations')
        for img_file in os.listdir(pat_image_path):
            json_file = img_file.replace('images', 'annotations') + '.json'
            try:
                with open(os.path.join(pat_annotations_path, json_file), 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
            except:
                print(json_file)
            lymph_modals = json_data['parts']["cervicalLymphNodes"]['perPart']['pictureModal']['tags']
            # 限制模态只能为超声
            # if 'us' not in lymph_modals:
            #     continue
            lymph_contours = json_data['parts']['cervicalLymphNodes']['perContour']['lymphContour']
            for lymph_contour in lymph_contours:
                tags = lymph_contour['tags']
               # 限制对恶性进行亚区分
                if 'malign' not in tags:
                    continue
                contour = lymph_contour['contour']
                contour_array = find_num(contour)
                box = cv2.boundingRect(contour_array)
                info = {
                    'image_name': os.path.join(hosp, pat, 'images', img_file),
                    'contour': contour,
                    'tags': tags,
                    'boxes': ','.join([str(i) for i in box])
                }
                infos.append(info)
    return infos


if __name__ == '__main__':
    dirs = '/mnt/data/hsy/数据/淋巴结分类数据/中山医院'
    hosp_list = ['SHSY', 'NTRM', 'JSZL', 'JSRM']

    all_fold_pkl = {
        'fold_1': {'train':[], 'val':[], 'test':[]},
        'fold_2': {'train':[], 'val':[], 'test':[]},
        'fold_3': {'train':[], 'val':[], 'test':[]},
        'fold_4': {'train':[], 'val':[], 'test':[]},
        'fold_5': {'train':[], 'val':[], 'test':[]}
    }


    for hosp in tqdm(hosp_list, ncols=100, desc='split data'):
        k_folds_list = split_K_fold(5, dirs, hosp)
        fold_pkl = generate_pkl(dirs, hosp, k_folds_list)
        for key in fold_pkl:
            for dataset_type in fold_pkl[key]:
                all_fold_pkl[key][dataset_type].extend(fold_pkl[key][dataset_type])

    with open('../refer/中山医院淋巴结malignNod_5_fold.pkl', 'wb') as fs:
        pickle.dump(all_fold_pkl, fs)







