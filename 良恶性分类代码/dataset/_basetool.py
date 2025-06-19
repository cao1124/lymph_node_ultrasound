import cv2
import random
import numpy as np
from typing import List, Tuple

class BoxTuneTool(object):
    # 实现对box的平移、缩放、旋转、翻转等操作
    def __init__(self,
                 box:List[int],
                 img_size:Tuple[int, int],
                 expand_pixel:int = 10):

        self.box = box
        self.img_size = img_size
        self.expand_pixel = expand_pixel

    def get_random_box(self,
                       window_size:Tuple[int, int]):
        # random chose a window and make sure the box is in the window

        x, y, w, h = self.box
        img_height, img_width = self.img_size
        window_height, window_width = self.img_size

        # find the max range
        x_max = min(img_width, x + window_width)
        y_max = min(img_height, y + window_height)

        # find the min range
        x_min = max(0, x + w - window_width)
        y_min = max(0, y + h - window_height)

        # random choose the box between the range

        x_left = random.uniform(
            x_min + self.expand_pixel, x_max - window_width - self.expand_pixel
        )
        y_top = random.uniform(
            y_min + self.expand_pixel, y_max - window_height - self.expand_pixel
        )

        x_right = min(x_left + window_width, img_width)
        y_bottom = min(y_top + window_height, img_height)

        return [int(x_left), int(y_top), int(x_right) - int(x_left), int(y_bottom) - int(y_top)]


    def get_center_box(self,
                       window_size:Tuple[int, int]):
        # chose the box center as the anchor point

        x, y, w, h = self.box
        x_center, y_center = x + w // 2, y + h // 2
        expand_pixel = [_ // 2 for _ in window_size]
        x_left = max(x_center - expand_pixel[1], 0)
        y_top = max(y_center - expand_pixel[0], 0)
        x_right = min(x_center + (window_size[1] - expand_pixel[1]), self.img_size[1])
        y_bottom = min(y_center + (window_size[0] - expand_pixel[0]), self.img_size[0])
        return [int(x_left), int(y_top), int(x_right) - int(x_left), int(y_bottom) - int(y_top)]

class BinaryBalanceSampler(object):

    def __init__(self,
                 strategy:str = 'down-sampling'):
        assert strategy in ['down-sampling', 'up-sampling', 'random-sampling'], 'The strategy must be down-sampling or up-sampling or random-sampling'
        self.method = strategy
        self.positive_num = 0
        self.negative_num = 0

    def __call__(self, train_data:List):

        negative_data = []
        positive_data = []
        for data in train_data:
            if 'malign' in data['tags']:
                self.negative_num += 1
                negative_data.append(data)
            elif 'benign' in data['tags']:
                self.positive_num += 1
                positive_data.append(data)
            else:
                raise ValueError('The tags must contain malign or benign')
        min_num = min(self.positive_num, self.negative_num)
        num_diff = abs(self.positive_num - self.negative_num)
        max_type = 'positive' if self.positive_num > self.negative_num else 'negative'
        if self.method == 'down-sampling':
            return self.down_sampling(negative_data, positive_data, min_num, max_type)
        elif self.method == 'up-sampling':
            return self.up_sampling(negative_data, positive_data, num_diff, max_type)

    def down_sampling(self, negative_data:List, positive_data:List, k:int, max_type:str):
        if max_type == 'positive':
            positive_data = random.sample(positive_data, k)
            return positive_data + negative_data
        elif max_type == 'negative':
            negative_data = random.sample(negative_data, k)
            return positive_data + negative_data
        else:
            raise ValueError('The max_type must be positive or negative')

    def up_sampling(self, negative_data:List, positive_data:List, k:int, max_type:str):
        if max_type == 'positive':
            extra_negative_data = random.choices(negative_data, k=k)
            return positive_data + extra_negative_data + negative_data
        elif max_type == 'negative':
            extra_positive_data = random.choices(positive_data, k=k)
            return positive_data + negative_data + extra_positive_data
        else:
            raise ValueError('The max_type must be positive or negative')

    def random_sampling(self):
        ...







