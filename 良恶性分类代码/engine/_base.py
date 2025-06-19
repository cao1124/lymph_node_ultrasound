import os.path
import sys
import random

class BaseEngine:

    def __init__(self, DEVICE):
        self.cuda = DEVICE

    def parse_data(self, **kwargs):
        # 解析数据
        raise NotImplementedError

    def predict_img(self, **kwargs):
        # 预测图片
        raise NotImplementedError

    def compute_metrics(self, **kwargs):
        # 计算指标
        raise NotImplementedError

    def save_img(self, **kwargs):
        # 保存图片
        raise NotImplementedError

    def grad_cam(self, **kwargs):
        # grad_cam
        raise NotImplementedError

    def load_model(self, **kwargs):
        # 加载模型
        raise NotImplementedError

