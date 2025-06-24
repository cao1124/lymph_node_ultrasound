#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：data_process.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/6/19 上午9:59 
"""
import os

from tqdm import tqdm


def paths_to_txt():
    output_file = r'F:\med_dataset\lymph淋巴结\中山淋巴结\20250623-中山淋巴恶性瘤淋巴瘤2分类-补充训练.txt'
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    # 确保扩展名是小写
    image_extensions = [ext.lower() for ext in image_extensions]

    with open(output_file, 'w', encoding='utf-8') as f:
        folder_path = r'F:\med_dataset\lymph淋巴结\中山淋巴结\训练补充-第二批肿瘤医院测试'
        for hospital_name in os.listdir(folder_path):
            for cls in os.listdir(os.path.join(folder_path, hospital_name)):
                if cls == '良性':
                    continue
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
                                f.write(file_path + ',' + '转移淋巴瘤\n')
                            elif cls == '非转移淋巴瘤':
                                f.write(file_path + ',' + '非转移淋巴瘤\n')
                            else:
                                print(file_path, ': 错误标签')
    print(f"已完成，共写入 {sum(1 for line in open(output_file, 'r', encoding='utf-8'))} 个图像路径到 {output_file}")


if __name__ == '__main__':
    paths_to_txt()
    print('done')
