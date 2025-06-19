import cv2
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from dataset.augumentations import get_transformations
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask


def cam_image_show_1(model:nn.Module, img_tensor:torch.FloatTensor, img:np.ndarray) -> Image.Image:
    with GradCAM(model, target_layer='features') as cam_extractor:
        pred_tensor = model(img_tensor)
        activation_map = cam_extractor(pred_tensor.squeeze(0).argmax().item(), pred_tensor)
        activation_map = activation_map[0].cpu().detach().numpy().squeeze(0)
        pil_img = to_pil_image(img)
        pil_map = to_pil_image(activation_map, mode='F')
        result = overlay_mask(pil_img, pil_map)
        return result


def cam_image_show2(model:nn.Module, pred_tensor:torch.FloatTensor, img:np.ndarray) -> np.ndarray:
    with GradCAM(model) as cam_extractor:
        activation_map = cam_extractor(pred_tensor.squeeze(0).argmax().item(), pred_tensor)
        activation_map = activation_map[0].cpu().detach().numpy().squeeze(0)
        activation_map = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
        activation_map = cv2.cvtColor(activation_map, cv2.COLOR_BGR2RGB)
        activation_map = cv2.resize(activation_map, (img.shape[1], img.shape[0]))
        cam = (1 - 0.6) * activation_map + 0.6 * img
        cam = cam / np.max(cam)
        cam = np.uint8(255 * cam)
        return cam







