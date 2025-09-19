#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：data_process.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/6/19 上午9:59 
"""
import json
import os
import shutil

import cv2
import numpy as np
from tqdm import tqdm

from classify_util import cv_read, cv_write


def paths_to_txt():
    output_file = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集txt\20250702-良恶性2分类-all.txt'
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # 确保扩展名是小写
    image_extensions = [ext.lower() for ext in image_extensions]

    with open(output_file, 'w', encoding='utf-8') as f:
        folder_path = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集'
        for hospital_name in os.listdir(folder_path):
            for cls in os.listdir(os.path.join(folder_path, hospital_name)):
                # if cls == '良性':
                #     continue
                for root, dirs, files in os.walk(os.path.join(folder_path, hospital_name, cls)):
                    if ',' in root:
                        os.rename(root, root.replace(',', ''))
                        root = root.replace(',', '')
                    print(root)
                    for file in tqdm(files):
                        if os.path.splitext(file)[1].lower() in image_extensions:
                            file_path = os.path.join(root, file)
                            if '报告' in file:
                                continue
                            if cls == '转移淋巴瘤':
                                f.write(file_path + ',' + '恶性\n')    # 转移淋巴瘤
                            elif cls == '非转移淋巴瘤':
                                f.write(file_path + ',' + '恶性\n')   # 非转移淋巴瘤
                            elif cls == '良性':
                                f.write(file_path + ',' + '良性\n')
                            else:
                                print(file_path, ': 错误标签')
    print(f"已完成，共写入 {sum(1 for line in open(output_file, 'r', encoding='utf-8'))} 个图像路径到 {output_file}")


def roi_crop_txt():
    roi_type = 'bbox'
    output_file = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集\20250625-中山淋巴恶性瘤淋巴瘤2分类-补充训练-crop.txt'
    ori_txt = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集\20250623-中山淋巴恶性瘤淋巴瘤2分类-补充训练-all.txt'
    img_dir = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集'
    out_lines = []
    with open(ori_txt, 'r', encoding='utf8') as lines:
        for line in tqdm(lines):
            # print(line)
            image_path = os.path.join(img_dir, line.split(',')[0])
            json_path = os.path.join(img_dir, line.split(',')[0]+'.json')
            if not os.path.exists(json_path):
                continue
            # 读取JSON文件
            with open(json_path, 'r', encoding='utf8') as f:
                data = json.load(f)
            # 获取淋巴结轮廓和模态标签
            lymph_contours = []
            try:
                # 从JSON结构中提取lymphContour
                contours = data['parts']['cervicalLymphNodes']['perContour']['lymphContour']
                for contour_data in contours:
                    lymph_contours.append({
                        'points': contour_data['contour'],
                        'tags': contour_data.get('tags', [])
                    })

                # 获取模态标签
                modal_tags = data['parts']['cervicalLymphNodes']['perPart']['pictureModal']['tags']
            except KeyError as e:
                print(f"JSON结构错误，缺少关键字段: {e}")
                continue

            # 检查模态是否为超声
            if 'us' not in [tag.lower() for tag in modal_tags]:
                # print("模态不是US，跳过处理")
                continue

            # 读取原始图像
            img = cv_read(image_path)
            if img is None:
                # print(f"无法读取图像: {image_path}")
                continue

            # 处理每个淋巴结轮廓
            for i, contour_data in enumerate(lymph_contours):
                # 解析轮廓点
                points = []
                for pair in contour_data['points'].split():
                    x, y = map(int, pair.split(','))
                    points.append([x, y])
                # 转换为NumPy数组
                contour = np.array(points, dtype=np.int32)
                if roi_type == 'mask':
                    # 创建掩膜
                    mask = np.zeros(img.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [contour], 255)
                    # 应用掩膜
                    roi = cv2.bitwise_and(img, img, mask=mask)
                elif roi_type == 'bbox':
                    # 计算最小外接矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    # 添加边界填充（10%）
                    pad_x = int(w * 0.1)
                    pad_y = int(h * 0.1)
                    # 计算裁剪区域（确保不超出图像边界）
                    x1 = max(0, x - pad_x)
                    y1 = max(0, y - pad_y)
                    x2 = min(img.shape[1], x + w + pad_x)
                    y2 = min(img.shape[0], y + h + pad_y)
                    # 裁剪矩形区域
                    roi = img[y1:y2, x1:x2]
                # 保存ROI图像
                if line.split('/')[0] == '江苏省肿瘤医院前期':
                    output_path = image_path.replace('江苏省肿瘤医院前期', '江苏省肿瘤医院前期-crop')
                    out_lines.append(line.replace('江苏省肿瘤医院前期', '江苏省肿瘤医院前期-crop'))
                elif line.split('/')[0] == '南通分类':
                    output_path = image_path.replace('南通分类', '南通分类-crop')
                    out_lines.append(line.replace('南通分类', '南通分类-crop'))
                elif line.split('/')[0] == '省人医所有淋巴结':
                    output_path = image_path.replace('省人医所有淋巴结', '省人医所有淋巴结-crop')
                    out_lines.append(line.replace('省人医所有淋巴结', '省人医所有淋巴结-crop'))
                elif line.split('/')[0] == '十院淋巴结筛选':
                    output_path = image_path.replace('十院淋巴结筛选', '十院淋巴结筛选-crop')
                    out_lines.append(line.replace('十院淋巴结筛选', '十院淋巴结筛选-crop'))
                elif line.split('/')[0] == '训练补充':
                    output_path = image_path.replace('训练补充', '训练补充-crop')
                    out_lines.append(line.replace('训练补充', '训练补充-crop'))
                elif line.split('/')[0] == '第二批肿瘤医院':
                    output_path = image_path.replace('第二批肿瘤医院', '第二批肿瘤医院-crop')
                    out_lines.append(line.replace('第二批肿瘤医院', '第二批肿瘤医院-crop'))
                else:
                    print(f"错误医院地址{line.split('/')[0]}")
                out_dir = os.path.dirname(output_path)
                os.makedirs(out_dir, exist_ok=True)
                cv_write(f'{os.path.splitext(output_path)[0]}.png', roi)
                # cv_write(f'{os.path.splitext(output_path)[0]}_roi_{i}_bbox.png', roi)
                # print(f"保存ROI图像: {output_path}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for ol in out_lines:
            f.write(ol)
    print(f"已完成，共写入 {len(out_lines)} 个图像路径到 {output_file}")


def roi_crop():
    roi_type = 'bbox'
    img_dir = r'F:\med_dataset\lymph淋巴结\中山淋巴结\域外测试集2\ori'
    for cls in os.listdir(img_dir):
        if cls == '良性':
            continue
        for root, dirs, files in os.walk(os.path.join(img_dir, cls)):
            for f in files:
                image_path = os.path.join(root, f)
                json_path = os.path.join(root, f+'.json')
                if not os.path.exists(json_path):
                    continue
                # 读取JSON文件
                with open(json_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                # 获取淋巴结轮廓和模态标签
                lymph_contours = []
                try:
                    # 从JSON结构中提取lymphContour
                    contours = data['parts']['cervicalLymphNodes']['perContour']['lymphContour']
                    for contour_data in contours:
                        lymph_contours.append({
                            'points': contour_data['contour'],
                            'tags': contour_data.get('tags', [])
                        })

                    # 获取模态标签
                    modal_tags = data['parts']['cervicalLymphNodes']['perPart']['pictureModal']['tags']
                except KeyError as e:
                    print(f"JSON结构错误，缺少关键字段: {e}")
                    continue

                # 检查模态是否为超声
                if 'us' not in [tag.lower() for tag in modal_tags]:
                    # print("模态不是US，跳过处理")
                    continue

                # 读取原始图像
                img = cv_read(image_path)
                if img is None:
                    # print(f"无法读取图像: {image_path}")
                    continue

                # 处理每个淋巴结轮廓
                for i, contour_data in enumerate(lymph_contours):
                    # 解析轮廓点
                    points = []
                    for pair in contour_data['points'].split():
                        x, y = map(int, pair.split(','))
                        points.append([x, y])
                    # 转换为NumPy数组
                    contour = np.array(points, dtype=np.int32)
                    if roi_type == 'mask':
                        # 创建掩膜
                        mask = np.zeros(img.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(mask, [contour], 255)
                        # 应用掩膜
                        roi = cv2.bitwise_and(img, img, mask=mask)
                    elif roi_type == 'bbox':
                        # 计算最小外接矩形
                        x, y, w, h = cv2.boundingRect(contour)
                        # 添加边界填充（10%）
                        pad_x = int(w * 0.1)
                        pad_y = int(h * 0.1)
                        # 计算裁剪区域（确保不超出图像边界）
                        x1 = max(0, x - pad_x)
                        y1 = max(0, y - pad_y)
                        x2 = min(img.shape[1], x + w + pad_x)
                        y2 = min(img.shape[0], y + h + pad_y)
                        # 裁剪矩形区域
                        roi = img[y1:y2, x1:x2]
                    # 保存ROI图像
                    output_path = image_path.replace('ori', 'crop')
                    out_dir = os.path.dirname(output_path)
                    os.makedirs(out_dir, exist_ok=True)
                    cv_write(f'{os.path.splitext(output_path)[0]}.png', roi)
                    # cv_write(f'{os.path.splitext(output_path)[0]}_roi_{i}_bbox.png', roi)
                    # print(f"保存ROI图像: {output_path}")


def txt_5cls():
    ori_txt = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集\20250623-中山淋巴恶性瘤淋巴瘤2分类-补充训练-all.txt'
    output_file = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集\20250625-中山转移淋巴瘤5分类.txt'
    out_lines = []
    path_files, name_files = [], []
    for ori_dir in [r'F:\med_dataset\lymph淋巴结\中山淋巴结\域外测试集1\ori\转移淋巴瘤',
                    r'F:\med_dataset\lymph淋巴结\中山淋巴结\域外测试集2\ori\转移淋巴瘤']:
        for root, dirs, files in os.walk(ori_dir):
            for f in files:
                if not f.endswith('.json'):
                    # f = os.path.splitext(f)[0] + '.png'
                    path_files.append(os.path.join(root, f))
                    name_files.append(f)
    with open(ori_txt, 'r', encoding='utf8') as lines:
        for line in tqdm(lines):
            # print(line)
            if '非转移淋巴瘤' in line:
                continue
            if '训练补充' in line:
                img_name = line.split(',')[0].split('/')[-1]
                idx = name_files.index(img_name)
                img_path = path_files[idx]
            else:
                img_path = line.split(',')[0]

            if '鼻' in img_path:
                ol = line.split(',')[0] + ',鼻咽'
            elif '肺' in img_path:
                ol = line.split(',')[0] + ',肺'
            elif '乳' in img_path:
                ol = line.split(',')[0] + ',乳腺'
            elif '食' in img_path:
                ol = line.split(',')[0] + ',食管'
            else:
                ol = line.split(',')[0] + ',其他'
            out_lines.append(ol)
    with open(output_file, 'w', encoding='utf-8') as f:
        for ol in out_lines:
            f.write(ol+'\n')
    print(f"已完成，共写入 {len(out_lines)} 个图像路径到 {output_file}")


def os_rename():
    folder_path = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练集\第二批肿瘤医院'
    for hospital_name in os.listdir(folder_path):
        for cls in os.listdir(os.path.join(folder_path, hospital_name)):
            if cls == '良性':
                continue
            for root, dirs, files in os.walk(os.path.join(folder_path, hospital_name, cls)):
                if ',' in root:
                    os.rename(root, root.replace(',', ''))


def generate_dataset_txt():
    all_lymph_root = r'E:\med_dataset\lymph淋巴结\中山淋巴结\弱标签数据\所有颈部淋巴'  # 包含10743张图
    swollen_lymph_root = r'E:\med_dataset\lymph淋巴结\中山淋巴结\弱标签数据\肿大颈部淋巴'  # 包含2543张图
    output_file = r'E:\med_dataset\lymph淋巴结\中山淋巴结\训练集txt\crop\20250812-良恶性肿大标签.txt'
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_extensions = [ext.lower() for ext in image_extensions]
    swollen_image_names = []
    for name in os.listdir(swollen_lymph_root):
        swollen_image_names.append(name)
    print(f"找到 {len(swollen_image_names)} 个肿大图像")

    # 写入输出文件
    with open(output_file, 'w', encoding='utf-8') as f:
        total_count = 0
        swollen_count = 0
        non_swollen_count = 0
        for name in tqdm(os.listdir(all_lymph_root), desc="处理图像"):
            if any(name.lower().endswith(ext) for ext in image_extensions):
                rel_path = os.path.join(all_lymph_root, name)
                if name in swollen_image_names:
                    swollen_label = '肿大'  # 对应 SwollenStatus.肿大 = 1
                    swollen_count += 1
                else:
                    swollen_label = '非肿大'  # 对应 SwollenStatus.非肿大 = 0
                    non_swollen_count += 1
                f.write(f"{rel_path},{swollen_label}\n")
                total_count += 1
    print(f"处理完成！总共写入 {total_count} 个图像路径")
    print(f"- 肿大图像: {swollen_count}")
    print(f"- 非肿大图像: {non_swollen_count}")
    print(f"文件已保存至: {output_file}")


import os


# 读取txt文件，获取所有图像文件名（不带路径）
def read_image_names_from_txt(txt_path):
    image_names = set()
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 提取图像文件名（路径的最后一部分）
            image_path = line.split(',')[0].strip()
            image_name = os.path.basename(image_path)
            image_names.add(image_name)
    return image_names


def filter_txt_by_existing_images(original_txt_path, image_dir, output_txt_path):
    """
    过滤txt文件，只保留在指定目录中实际存在的图像路径

    参数:
    original_txt_path: 原始txt文件路径
    image_dir: 图像目录路径
    output_txt_path: 输出txt文件路径
    """
    # 读取原始txt文件
    with open(original_txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 创建输出文件
    with open(output_txt_path, 'w', encoding='utf-8') as out_f:
        for line in lines:
            # 提取图像路径和标签
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue

            image_path = parts[0]
            label = parts[1]

            # 从路径中提取文件名
            filename = os.path.basename(image_path)

            # 构建完整的图像文件路径
            full_image_path = os.path.join(image_dir, filename)

            # 检查文件是否存在
            if os.path.exists(full_image_path):
                out_f.write(line)
                print(f"保留: {filename}")
            else:
                print(f"跳过: {filename} (文件不存在)")


def train_data_shutil():
    src_root = r"E:\med_dataset\lymph淋巴结\中山淋巴结\训练集"
    dst_root = r"E:\med_dataset\lymph淋巴结\中山淋巴结\训练集-待标注"

    for root, _, files in os.walk(src_root):
        for fname in files:
            if fname.lower().endswith(".png"):
                img_path = os.path.join(root, fname)
                json_path = img_path + ".json"
                if not os.path.exists(json_path):
                    # 计算相对路径（去掉源根目录）
                    rel_path = os.path.relpath(img_path, src_root)
                    dst_path = os.path.join(dst_root, rel_path)

                    # 创建目标目录
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                    # 复制文件
                    shutil.copy2(img_path, dst_path)
                    print(f"Copied: {img_path} -> {dst_path}")
    print("Done: 未带标注的图像已提取到 训练集-待标注 路径下。")


if __name__ == '__main__':
    # paths_to_txt()
    # roi_crop()
    # roi_crop_txt()
    # txt_5cls()
    # os_rename()
    # generate_dataset_txt()
    
    # 训练集-待标注
    train_data_shutil()
    
    # original_txt = "/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/crop/20250910-中山淋巴恶性瘤淋巴瘤2分类-补充训练-all.txt"
    # image_dir = "/mnt/disk1/caoxu/dataset/中山淋巴结/训练集crop"
    # output_txt = "/mnt/disk1/caoxu/dataset/中山淋巴结/训练集txt/crop/20250910-filtered_dataset.txt"
    #
    # filter_txt_by_existing_images(original_txt, image_dir, output_txt)
    # print(f"过滤完成! 结果保存在 {output_txt}")

    # # 计算重复文件
    # all_lymph_root = r'E:\med_dataset\lymph淋巴结\中山淋巴结\弱标签数据\所有颈部淋巴'  # 包含10743张图
    # swollen_lymph_root = r'E:\med_dataset\lymph淋巴结\中山淋巴结\弱标签数据\肿大颈部淋巴'  # 包含2543张图
    # all_lymph_files = set(os.listdir(all_lymph_root))
    # swollen_lymph_files = set(os.listdir(swollen_lymph_root))
    # common_files = all_lymph_files.intersection(swollen_lymph_files)
    # # 输出重复文件数量
    # print(f"重复的文件数量: {len(common_files)}")

    print('done')
