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

import torch
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from tqdm import tqdm
from classify_util import *


def test_image():
    # for i in range(4):
    # cls_model = f'model/20250623-中山淋巴恶性瘤淋巴瘤2分类-resnet50-bs200-lr0.0001/fold{i}-best-acc-model.pt'
    cls_model = 'model/20250702-良恶性2分类-原图0.8591.pt'
    print(cls_model)
    cls_model = torch.load(cls_model, map_location=device)
    cls_model = modelMap(cls_model, device).eval()

    base_dir = r'E:\med_dataset\lymph淋巴结\中山淋巴结\域外测试集2\ori'
    pres, labels = [], []
    for cls in os.listdir(base_dir):
        # if cls == '良性':
        #     continue
        for root, dirs, files in os.walk(os.path.join(base_dir, cls)):
            print(root)
            for f in tqdm(files):
                if is_image(os.path.join(root, f)):
                    img_ori = cv_read(os.path.join(root, f), flag=3)
                    img = Image.fromarray(img_ori).convert('RGB')
                    img = trans(img)
                    img = torch.unsqueeze(img, dim=0).to(device)

                    with torch.no_grad():
                        res, _ = sec_pre(img, cls_model, device)
                        pres.append(LymphPathologicCls2CN(res).name)
                        if cls == '良性':
                            labels.append('良性')
                        else:
                            labels.append('恶性')
    print('confusion_matrix:\n{}'.format(confusion_matrix(labels, pres)))
    print('classification_report:\n{}'.format(classification_report(labels, pres, digits=4)))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean, std = [0.29003, 0.29385, 0.31377], [0.18866, 0.19251, 0.19958]
    trans = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean, std)])
    test_image()
    print('Done')
