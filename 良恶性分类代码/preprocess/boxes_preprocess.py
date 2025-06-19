import cv2
import os
import random
import json
import numpy as np
from tqdm import tqdm
from about_crop import mask2box, crop_patch_1, crop_patch_2, resize_img_keep_ratio
from typing import List, Tuple
import pickle

boundary_distance=0
nodules_size_file= '../refer/all_nodule_statistics_res_0802/nodules_size.json'
with open(nodules_size_file, 'r', encoding='utf8')as fp:
    nodules_size = json.load(fp)

def modified_get_box_main(domain_dir, domain_list, res_dir):
    data = {}
    for domain in domain_list:
        domain_nodule_data = []
        domain_normal_data = []
        domain_benign_num = 0
        domain_path = os.path.join(domain_dir, domain)
        patient_list = os.listdir(domain_path)
        for patient in tqdm(patient_list, desc=f'数据预处理_{domain}'):
            patient_dir = os.path.join(domain_path, patient)
            images_dir = os.path.join(patient_dir,'images')
            masks_file = 'modified_masks' if os.path.exists(os.path.join(patient_dir, 'modified_masks')) else 'masks'
            masks_dir = os.path.join(patient_dir, masks_file)
            nod_boxes, normal_boxes, benign_num = new_get_nodule_boxes_and_random_crop_normal(
                hosp=domain,
                images_dir=images_dir,
                masks_dir=masks_dir
            )
            domain_nodule_data.extend(nod_boxes)
            domain_normal_data.extend(normal_boxes)
            domain_benign_num += benign_num

        # 再对 normal 样本进行一次筛选，满足数量要求

        if len(domain_normal_data) > domain_benign_num:
            domain_normal_data = random.sample(domain_normal_data, domain_benign_num)
        data[domain] = domain_nodule_data + domain_normal_data
        print('domain：%s; 数据量：%d' % (domain, len(data[domain])))

    res_path = os.path.join(res_dir, f'modified_boxes_info.pkl')
    with open(res_path, 'wb') as fs:
        pickle.dump(data, fs)

    res_path = os.path.join(res_dir, f'modified_boxes_info.json')
    with open(res_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4, separators=(',', ':'))

def new_get_nodule_boxes_and_random_crop_normal(hosp:str, images_dir:str, masks_dir:str):
    nod_boxes = [] # 良恶性结节
    normal_boxes = [] # 正常样本
    benign_num = 0
    for mask_name in os.listdir(masks_dir):
        mask_path = os.path.join(masks_dir, mask_name)
        img_path = os.path.join(images_dir, mask_name)
        if not os.path.exists(img_path):
            try:
                img_path = img_path.replace('png', 'jpg')
            except:
                raise TypeError('The picture must end with jpg or png')

        save_img_path_part = ('/').join(img_path.split('/')[-4:])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # img
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # img

        img_h, img_w = img.shape[:2]
        img_size = (img_h, img_w)
        benign_mask = np.zeros((img_h, img_w), dtype=np.uint8)
        malig_mask = np.zeros((img_h, img_w), dtype=np.uint8)

        benign_mask[mask == 128] = 128  # benign, 良性的灰度像素为128
        malig_mask[mask == 255] = 255  # malig, 恶性的灰度像素为256

        benign_boxes = mask2box(benign_mask)
        malig_boxes = mask2box(malig_mask)

        benign_num = len(benign_boxes)

        # 添加结节中的良性样本
        if benign_boxes:
            for id, box in enumerate(benign_boxes):
                nod_boxes.append(['良性', box, hosp, save_img_path_part])

        # 添加结节中的恶性样本
        if malig_boxes:
            for id, box in enumerate(malig_boxes):
                nod_boxes.append(['恶性', box, hosp, save_img_path_part])

        # 添加正常样本
        nodule_num = len(nod_boxes)
        if nodule_num > 0:
            crop_size = random.sample(nodules_size, 1)[0]
            if nodule_num > 1:
                i = random.randint(0, nodule_num - 1)
                boxes = nod_boxes[i]
            else:
                boxes = nod_boxes[0]

            normal_box, roi = random_crop_normal(boxes[1], img_size, crop_size, ratio=0.15, boundary_distance=20)
            if normal_box is not None:
                normal_boxes.append(['正常',normal_box,hosp, save_img_path_part])
    return nod_boxes, normal_boxes, benign_num

def random_crop_normal(boxes: List[int],
                       img_size: Tuple[int, int], crop_size:dict,
                       ratio:float, boundary_distance:int):
    try:
        w,  h = crop_size.get('w'), crop_size.get('h')
    except:
        raise KeyError('check the key')
    box_info = []
    x, y, w_b, h_b = boxes
    x_min, y_min = x - w_b // 2, y - h_b // 2
    x_max, y_max = x + w_b // 2, y + h_b // 2

    expand_direction = {0:'left_top', 1:'left_down',
                        2:'right_top', 3:'right_down'}
    index = 0
    for x in (x_min, x_max):
        for y in (y_min, y_max):
            points = (x, y)
            crop_size = (w, h)
            expand_pixel = int(max(crop_size) * ratio)
            x_new, y_new = crop_path(points, crop_size, expand_pixel, boundary_distance, expand_direction[index])
            index += 1
            # 如果生成的框越界了, 则不将其作为候选对象
            if (x_new - w // 2) <= 0 or (x_new + w // 2) >= img_size[1] or (y_new - h // 2) <= 0 or (y_new + h // 2) >= img_size[0]:
                continue
            else:
                box_info.append([x_new, y_new])
    if box_info:
        i = np.random.randint(0, len(box_info))
        x_new, y_new = box_info[i]
        dim1_cut_min = max(0, y_new - expand_pixel - h // 2)
        dim1_cut_max = min(img_size[0] - 1, y_new + h // 2 + expand_pixel)
        dim2_cut_min = max(0, x_new - expand_pixel - w // 2)
        dim2_cut_max = min(img_size[1] - 1, x_new + w // 2 + expand_pixel)
        roi = [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]
        box = [x_new, y_new, w, h]
    else:
        box = None
        roi = None
    return box, roi

def crop_path(points:Tuple[int, int], crop_size:Tuple[int, int], expand_pixel:int, boundary_distance:int, expand_direction:str):
    x_0, y_0 = points
    if expand_direction == 'left_top':
        # 限制只能在框的左上角进行截取
        x_new = random.randint(max(x_0 - crop_size[1] - boundary_distance, 0), max(x_0 - crop_size[1], 0))
        y_new = random.randint(max(y_0 - crop_size[0] - boundary_distance, 0), max(y_0 - crop_size[0], 0))
    elif expand_direction == 'right_top':
        # 限制只能在右上角截取
        x_new = random.randint(x_0 + boundary_distance, x_0 + crop_size[1] + boundary_distance)
        y_new = random.randint(max(y_0 - crop_size[0] - boundary_distance, 0), max(y_0 - crop_size[0], 0))
    elif expand_direction == 'left_down':
        # 限制只能从左下角截取
        x_new = random.randint(max(x_0 - crop_size[1] - boundary_distance , 0), max(x_0 - crop_size[1], 0))
        y_new = random.randint(y_0 + boundary_distance , y_0 + crop_size[0] + boundary_distance)
    elif expand_direction == 'right_down':
        # 限制只能从右下角截取
        x_new = random.randint(x_0 + boundary_distance, x_0 + crop_size[1] + boundary_distance)
        y_new = random.randint(y_0 + boundary_distance, y_0 + crop_size[0] + boundary_distance)
    else:
        raise NameError
    return x_new, y_new




if __name__ == '__main__':
    domain_dir = '../data/thyroid_data_desensitized/all/organized_imgs'
    domain_list="hx_yh_zk_mnw_gk_sh10_scsrm".split('_')
    boxes_res_dir = '../data/thyroid_data_desensitized/all/extracted_boxes'
    modified_get_box_main(domain_dir, domain_list, boxes_res_dir)