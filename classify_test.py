#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@Project ：lymph_node_ultrasound 
@File    ：classify_test.py
@IDE     ：PyCharm 
@Author  ：cao xu
@Date    ：2025/6/20 上午9:05 
"""
import os

import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from tqdm import tqdm
from classify_util import modelMap, is_image, cv_read, sec_pre, LymphCls2CN, LymphClsTrans2CN


def test_image():
    cls_model = 'model/20250619-中山淋巴恶性瘤淋巴瘤2分类-0.8913.pt'
    # cls_model = 'model/20250625-中山淋巴恶性瘤淋巴瘤2分类-0.8108.pt'
    cls_model = torch.load(cls_model, map_location=device)
    cls_model = modelMap(cls_model, device).eval()

    base_dir = r'E:\med_dataset\lymph淋巴结\中山淋巴结\20251105-内外部测试-人机对比\外部验证\中山一人一图\恶性'
    pres, labels = [], []
    for cls in tqdm(os.listdir(base_dir)):
        if cls == '良性':
            continue
        for root, dirs, files in os.walk(os.path.join(base_dir, cls)):
            # print(root)
            for f in files:
                if is_image(os.path.join(root, f)):
                    img_ori = cv_read(os.path.join(root, f), flag=3)
                    img = Image.fromarray(img_ori).convert('RGB')
                    img = trans(img)
                    img = torch.unsqueeze(img, dim=0).to(device)

                    with torch.no_grad():
                        res, _ = sec_pre(img, cls_model, device)
                        pres.append(LymphClsTrans2CN(res).name)
                        if cls == '转移':
                            labels.append('转移淋巴瘤')
                        else:
                            labels.append('非转移淋巴瘤')
    print('confusion_matrix:\n{}'.format(confusion_matrix(labels, pres)))
    print('classification_report:\n{}'.format(classification_report(labels, pres, digits=4)))


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    # trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    # test_image()

    excel_path = r'D:\med_code\lymph_node_ultrasound\plot_util\转移非转移分类-外部验证结果.xlsx'
    excel_data = pd.read_excel(excel_path, sheet_name='人机对比')
    labels = excel_data['label'].tolist()
    pred = excel_data['AI'].tolist()

    # confusion_matrix
    print('confusion_matrix:\n{}'.format(confusion_matrix(labels, pred)))
    print('classification_report:\n{}'.format(classification_report(labels, pred, digits=4)))
    print('Done')
