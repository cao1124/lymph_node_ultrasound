# 中山的淋巴结数据处理
import copy
import os
import re
import cv2
import random
import pickle
import numpy as np
import json
from tqdm import tqdm
from util import find_num, json2mm

def func():
    dirs = '/mnt/data/hsy/科研项目/良恶性分类代码/filter_data/上海中山恶性亚分分型'
    pkl_info = {}
    for hosp in os.listdir(dirs):
        pkl_info[hosp] = {}
        hosp_dir = os.path.join(dirs, hosp)
        pat_list = os.listdir(hosp_dir)
        random.seed(101)
        random.shuffle(pat_list)
        train_val_test = {
            'train': pat_list[:int(len(pat_list)*0.7)],
            'val': pat_list[int(len(pat_list)*0.7):int(len(pat_list)*0.9)],
            'test': pat_list[int(len(pat_list)*0.9):]
        }
        for dataset_type in train_val_test:
            pkl_info[hosp][dataset_type] = []
            for pat in tqdm(train_val_test[dataset_type], desc=f'preprocessing:{hosp}_{dataset_type}', ncols=100):
                pat_image_path = os.path.join(hosp_dir, pat, 'images')
                pat_annotations_path = os.path.join(hosp_dir, pat, 'annotations')
                for img_file in os.listdir(pat_image_path):
                    json_file = img_file.replace('images', 'annotations') + '.json'
                    try:
                        with open(os.path.join(pat_annotations_path, json_file), 'r', encoding='utf-8') as f:
                            json_data = json.load(f)
                    except:
                        print(json_file)
                    mod = json_data['parts']["cervicalLymphNodes"]['perPart']['pictureModal']['tags']
                    # if 'us' not in mod:
                    #     continue
                    lymph_contours = json_data['parts']['cervicalLymphNodes']['perContour']['lymphContour']
                    for lymph_contour in lymph_contours:
                        tags = lymph_contour['tags']
                        if ('transfer' in tags or 'lymphomas' in tags) and 'malign' in tags:
                            contour = lymph_contour['contour']
                            contour_array = find_num(contour)
                            box = cv2.boundingRect(contour_array)
                            info = {
                                'image_name': os.path.join(hosp, pat, 'images', img_file),
                                'contour': contour,
                                'tags': lymph_contour['tags'],
                                'boxes': ','.join([str(i) for i in box])
                            }
                            pkl_info[hosp][dataset_type].append(info)
    with open('../refer/中山医院淋巴结数据_恶性亚分型_筛选后.pkl', 'wb') as fs:
        pickle.dump(pkl_info, fs)

def generate_mask():
    # 根据中山的annotation标注生成mask
    dir = '/mnt/data/hsy/数据/淋巴结检测数据/中山医院科研数据'
    for hosp in os.listdir(dir):
        hosp_dir = os.path.join(dir, hosp)
        for pat in tqdm(os.listdir(hosp_dir), desc=f'generating mask for {hosp}', ncols=100):
            pat_image_dir = os.path.join(hosp_dir, pat, 'images')
            pat_json_dir = os.path.join(hosp_dir, pat, 'annotations')
            pat_mask_dir = os.path.join(hosp_dir, pat, 'masks')
            os.makedirs(pat_mask_dir, exist_ok=True)
            for file in os.listdir(pat_json_dir):
                image_name = file[:-5]
                json_path = os.path.join(pat_json_dir, file)
                image_path = os.path.join(pat_image_dir, image_name)
                mask, _ = json2mm(json_path, image_path, ['cervicalLymphNodes', 'perContour', 'lymphContour'])
                cv2.imwrite(os.path.join(pat_mask_dir, image_name), mask)

if __name__ == '__main__':
    import pickle

    domain_dir = '/mnt/data/hsy/科研项目/良恶性分类代码/filter_data/上海中山'
    with open('../refer/中山医院淋巴结数据_一致性筛选.pkl', 'rb') as f:
        data = pickle.load(f)
    copy_data = copy.deepcopy(data)
    new_data = {}

    # 重新修改数据
    for hosp in copy_data:
        new_data[hosp] = {}
        hosp_data = copy_data[hosp]
        for type in ['train', 'val', 'test']:
            new_data[hosp][type] = []
            type_data = hosp_data[type]
            for each_data in tqdm(type_data, ncols=100):
                image_name = each_data['image_name']
                contour = each_data['contour']
                annotation_name = image_name.replace('images', 'annotations') + '.json'

                with open(os.path.join(domain_dir, annotation_name), 'r', encoding='utf-8', errors='ignore') as fj:
                    annotations_data = json.load(fj)


                lymph_infos = annotations_data['parts']['cervicalLymphNodes']['perContour']['lymphContour']
                for lymph_info in lymph_infos:
                    lymph_contour = lymph_info['contour']
                    if lymph_contour == contour:
                        if 'malign' in lymph_info['tags']:
                            each_data['tags'] = lymph_info['tags']
                            new_data[hosp][type].append(each_data)
                        else:
                            continue
                    else:
                        continue
    with open('../refer/中山医院淋巴结数据_一致性筛选_恶性有亚分型信息.pkl', 'wb') as fss:
        pickle.dump(new_data, fss)

    with open('../refer/中山医院淋巴结数据_一致性筛选_恶性有亚分型信息.pkl', 'rb') as f:
        data = pickle.load(f)

    transfer = 0
    lymphos = 0

    for hosp in data:
        hosp_data = data[hosp]
        for type in ['train', 'val', 'test']:
            type_hosp_data = hosp_data[type]
            for each_data in type_hosp_data:
                tags = each_data['tags']
                if 'transfer' in tags:
                    transfer += 1
                if 'lymphomas' in tags:
                    lymphos += 1
    print(transfer, lymphos)







