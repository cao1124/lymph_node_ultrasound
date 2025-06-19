import os
import random
import re

import cv2
import pickle
from collections import defaultdict
from tqdm import tqdm
from util import read_json

def physiological_test_data(root_dir, generate_train_val=False):
    data_pkl = defaultdict(list)
    for domain in os.listdir(root_dir):
        data_pkl[domain] = []
        domain_dir = os.path.join(root_dir, domain)
        for pat in tqdm(os.listdir(domain_dir), ncols=100):
            pat_dir = os.path.join(domain_dir, pat)
            pat_annotations_dir = os.path.join(pat_dir, 'annotations')
            if not os.path.exists(pat_annotations_dir):
                continue
            for file in os.listdir(pat_annotations_dir):
                file_path = os.path.join(pat_annotations_dir, file)
                data_info = read_json(file_path)

                lymph_infos = data_info["parts"]["cervicalLymphNodes"]["perContour"]["lymphContour"]
                if len(lymph_infos) > 0:
                    for lymph_info in lymph_infos:
                        tags = lymph_info["tags"]
                        boxes = re.findall(r'\d+', lymph_info['boxes'])
                        _, _, w, h = [int(i) for i in boxes]
                        if w < 256 and h < 256:
                            continue
                        else:
                            if 'PathologicalJudgment' in tags:
                                data_pkl[domain].append(
                                    {
                                        'image_name':os.path.join(domain, pat, 'images', file[:-5]),
                                        'contour':lymph_info['contour'],
                                        'tags':tags,
                                        'boxes':lymph_info['boxes']
                                    }
                                )

    print(data_pkl)
    with open('../refer/physiological_test_data_>256.pkl', 'wb') as fs:
        pickle.dump(data_pkl, fs)

    if generate_train_val:
        # 将这批数据按比例划分为训练集和验证集
        physiological_train_val_pkl = defaultdict(dict)

        for domain in data_pkl:
            data = data_pkl[domain]
            domain_pat = list(set([pat['image_name'].split('/')[1] for pat in data]))
            random.seed(101)
            random.shuffle(domain_pat)

            train_pat = domain_pat[:int(len(domain_pat)*0.8)]
            val_path = domain_pat[int(len(domain_pat)*0.8):]

            train_data = [info for info in data if info['image_name'].split('/')[1] in train_pat]
            val_data = [info for info in data if info['image_name'].split('/')[1] in val_path]

            physiological_train_val_pkl[domain]['train'] = train_data
            physiological_train_val_pkl[domain]['val'] = val_data
        with open('../refer/physiological_train_val_data.pkl', 'wb') as fs:
            pickle.dump(physiological_train_val_pkl, fs)

def getfilePkl(root_dir:str, domain:str, dataset_type:str='test') -> dict:
    # 读取一张图像的json数据, 生成对应的pkl文件

    domain_dir = os.path.join(root_dir, domain)

    domain_pkl = defaultdict(dict)
    domain_pkl[domain] = defaultdict(list)

    for pat in os.listdir(domain_dir):
        pat_annotations_path = os.path.join(domain_dir, pat, 'annotations')
        for file in os.listdir(pat_annotations_path):
            file_path = os.path.join(pat_annotations_path, file)
            data_info = read_json(file_path)
            lymph_infos = data_info["parts"]["cervicalLymphNodes"]["perContour"]["lymphContour"]
            if len(lymph_infos) > 0:
                for lymph_info in lymph_infos:
                    tags = lymph_info["tags"]
                    if 'PathologicalJudgment' in tags:
                        data_pkl = {
                            'image_name':os.path.join(domain, pat, 'images', file[:-5]),
                            'contour':lymph_info['contour'],
                            'true_label':tags,
                            'gt_box':lymph_info['boxes']
                        }

                        domain_pkl[domain][dataset_type].append(data_pkl)
    return domain_pkl




if __name__ == '__main__':
    import torch
    import torch.nn as nn


    class MyModule(nn.Module):
        def __init__(self):
            super(MyModule, self).__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1, bias=False)
            self.conv2 = nn.Conv2d(in_channels=6, out_channels=9, kernel_size=3, stride=1, padding=1, bias=False)

            self.register_buffer('weight', torch.ones(10, 10))
            self.register_buffer('bias', torch.zeros(10))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = x * self.weight + self.bias
            return x


    net = MyModule()

    for name, param in net.named_parameters():
        print(name, param.shape)

    print('\n', '*' * 40, '\n')

    for key, val in net.state_dict().items():
        print(key, val.shape)