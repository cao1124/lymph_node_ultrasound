import os
import cv2
import pickle
from tqdm import tqdm
from util import mask2box
from collections import  defaultdict
# ================================================== 划分数据集 ==========================================================

def main():
    root_dirs = '/mnt/data/hsy/数据/淋巴结检测数据/华西彭老师科研数据/转移非转移分类'
    tags = ['转移', '良性']
    hosp = 'HX'
    dicts = {}
    dicts[hosp] = defaultdict(list)

    for tag in tags:
        label = 'benign' if tag == '良性' else 'malign'
        tag_dirs = os.path.join(root_dirs, tag)
        if tag == '转移':
            tag_image_dirs = os.path.join(tag_dirs, '转移原始图像')
            tag_mask_dirs = os.path.join(tag_dirs, '转移图像mask')
        else:
            tag_image_dirs = os.path.join(tag_dirs, '良性淋巴结原始图像')
            tag_mask_dirs = os.path.join(tag_dirs, '良性淋巴结mask')

        image_files = os.listdir(tag_image_dirs)
        file_len = len(image_files)

        train_val_test = \
            {
                'train': image_files[:int(file_len*0.7)],
                'val': image_files[int(file_len*0.7):int(file_len*0.9)],
                'test': image_files[int(file_len*0.9):]
            }

        for dataset_type in train_val_test:
            type_image_files = train_val_test[dataset_type]

            for each_image_file in tqdm(type_image_files, ncols=100):

                image_path = os.path.join(tag_image_dirs, each_image_file)
                mask_path = os.path.join(tag_mask_dirs, each_image_file)

                boxes, contours = mask2box(mask_path, type='xywh')

                for contour, box in zip(contours, boxes):
                    contour_list = contour.reshape(-1, 2).tolist()
                    infos = \
                        {
                            'image_name': image_path,
                            'contour': ' '.join(f'{i[0]},{i[1]}' for i in contour_list),
                            'tags': label,
                            'boxes': ','.join([str(i) for i in box]),
                        }
                    dicts[hosp][dataset_type].append(infos)

    with open('../refer/HX_lymph_data.pkl', 'wb') as fs:
        pickle.dump(dicts, fs)













