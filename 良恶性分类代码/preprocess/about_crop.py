import os
import cv2
import  numpy as np
import time
import pandas as pd
from tqdm import tqdm
import shutil
import random

# 结节区域外扩
def crop_roi_extend(box, img, ratio=0.15, size=()):
    x, y, w, h = box

    max_length = max(w, h)
    expand_pixel = int(max_length * ratio)

    dim1_cut_min = max(0, y - expand_pixel)
    dim1_cut_max = min(size[0] - 1, y + h + expand_pixel)
    dim2_cut_min = max(0, x - expand_pixel)
    dim2_cut_max = min(size[1] - 1, x + w + expand_pixel)
   
    roi = [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]
    return img[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max], roi

def scale_box(box, ratio=0.15, size=[]):
    x, y, w, h = box
    if ratio >= 0:
        w_ex_pixel = int((w * ratio) // 2)
        h_ex_pixel = int((h * ratio) // 2)
        dim2_cut_min = max(0, x - w_ex_pixel)
        dim2_cut_max = min(size[1] - 1, x + w + w_ex_pixel)
        dim1_cut_min = max(0, y - h_ex_pixel)
        dim1_cut_max = min(size[0] - 1, y + h + h_ex_pixel)
        new_box = [dim2_cut_min, dim1_cut_min, dim2_cut_max-dim2_cut_min, dim1_cut_max-dim1_cut_min]
        return new_box
    else:
        w_shrink_pixel = int((w * abs(ratio)) // 2)
        h_shrink_pixel = int((h * abs(ratio)) // 2)
        dim2_cut_min = x + w_shrink_pixel
        dim2_cut_max = x + w - w_shrink_pixel
        dim1_cut_min = y + h_shrink_pixel
        dim1_cut_max = y + h - h_shrink_pixel
        if dim2_cut_min>=dim2_cut_max or dim1_cut_min>=dim1_cut_max:
            return box
        else:
            new_box = [dim2_cut_min, dim1_cut_min, dim2_cut_max-dim2_cut_min, dim1_cut_max-dim1_cut_min]
            return new_box


def shiftXY(box, shift_range, size):
    x, y, w, h = box
    shiftx = int(w*np.random.uniform(shift_range[0], shift_range[1], 1))
    shifty = int(h*np.random.uniform(shift_range[0], shift_range[1], 1))
    
    new_x = x+shiftx if x+shiftx >= 0 else 0
    new_x = x+shiftx if x+shiftx < x+w else x
    new_y = y+shifty if y+shifty >= 0 else 0
    new_y = y+shifty if y+shifty < y+h else y

    if new_x + w > size[1]:
        w = size[1] - new_x
    if new_y + h > size[0]:
        h = size[0] - new_y
    box = [new_x, new_y, w , h]
    return box


def extendXY(box, extend_xy_ratio_range, size):
    x, y, w, h = box
    ext_x = int(w*np.random.uniform(extend_xy_ratio_range[0], extend_xy_ratio_range[1], 1))
    ext_y = int(h*np.random.uniform(extend_xy_ratio_range[0], extend_xy_ratio_range[1], 1))
    box = [box[0], box[1], box[2] + x , box[3] + y]

    if w+ext_x > 0 and x+w+ext_x < size[1]:
        w = w+ext_x
    if h+ext_y > 0 and y+h+ext_y < size[0]:
        h = h+ext_y

    box = [x, y, w , h]
    return box

# 保持长宽比缩放图像
def resize_img_keep_ratio(img, target_size=[224,224]):
    old_height, old_width = img.shape[:2]
    ratio = min(target_size[0] / old_height, target_size[1] / old_width)
    new_height = int(old_height * ratio + 0.5)
    new_width = int(old_width * ratio + 0.5)
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

    pad_height = target_size[0] - new_height
    pad_width = target_size[1] - new_width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left

    img_new = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value=[0, 0, 0])
    return img_new


def mask2box(mask):
    # gray = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    gray = mask
    ret, binary = cv2.threshold( gray, 0, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
 
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        boxes.append([x,y,w,h])
    return boxes


def draw_cont(original_image, mask_image):
    # 阈值处理，将物体区域转换为二值图像
    _, binary_image = cv2.threshold(mask_image, 1, 255, cv2.THRESH_BINARY)
    # 轮廓检测
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 在原图上绘制轮廓线
    contour_image = np.copy(original_image)
    cv2.drawContours(contour_image, contours, -1, (0, 0, 255), 2)  # 使用红色线条绘制轮廓线，线宽为 2
    # 保存绘制了轮廓线的原图
    return contour_image
   

#对单个图片随机生成正常结节的crop区域
def crop_patch_1(img_size, crop_size, ratio, boundary_distance):
    w, h = crop_size["w"], crop_size["h"]
    max_length = max(w, h)
    expand_pixel = int(max_length * ratio)

    y_0 = random.randint(boundary_distance + expand_pixel, img_size[0]-boundary_distance - expand_pixel)
    x_0 = random.randint(boundary_distance + expand_pixel, img_size[1]-boundary_distance - expand_pixel)   

    dim1_cut_min = max(0, y_0 - expand_pixel)
    dim1_cut_max = min(img_size[0] - 1, y_0 + h + expand_pixel)
    dim2_cut_min = max(0, x_0 - expand_pixel)
    dim2_cut_max = min(img_size[1] - 1, x_0 + w + expand_pixel)

    roi = [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max] #(左上角x坐标, 右下角x坐标, 左上角y坐标, 右下角y坐标)
    box = [x_0, y_0, w, h] #(框的中心坐标 + 框的宽高)
    #return mask[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max], roi    
    return roi, box

def crop_patch_new(boxes, img_size, crop_size, ratio, boundary_distance):
    w, h = crop_size["w"], crop_size["h"]
    max_length = max(w, h)
    expand_pixel = int(max_length * ratio)

    # 对截取的区域进行限制
    # 首先获得结节框的左上角坐标和右下角坐标
    x, y , w_b,  h_b = boxes
    x_min, y_min = x - w_b // 2, y - h_b // 2
    x_max, y_max = x + (w_b - w_b // 2), y + (h_b - h_b // 2)

    # 通过左上角坐标和右下角坐标对区域进行限制

    img_y_max = min(y_max + 1 * h, img_size[0])
    img_x_max = min(x_max + 1 * w, img_size[1])
    img_y_min = max(y_min - 1 * h,  0)
    img_x_min = max(x_min - 1 * w,  0)

    # 然后随机选择中心点

    # 为了防止boundary_distance + expand_pixel 过大，导致随机选择的中心点无法生成crop区域
    y_limit = max(img_y_min + boundary_distance + expand_pixel + 1, img_y_max - boundary_distance - expand_pixel)
    x_limit = max(img_x_min + boundary_distance + expand_pixel + 1, img_x_max - boundary_distance - expand_pixel)

    y_0 = random.randint(img_y_min + boundary_distance + expand_pixel, y_limit)
    x_0 = random.randint(img_x_min + boundary_distance + expand_pixel, x_limit)

    dim1_cut_min = max(img_y_min, y_0 - expand_pixel)
    dim1_cut_max = min(img_y_max - 1, y_0 + h + expand_pixel)
    dim2_cut_min = max(img_x_min, x_0 - expand_pixel)
    dim2_cut_max = min(img_x_max - 1, x_0 + w + expand_pixel)

    roi = [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]
    box = [x_0, y_0, w, h]
    return roi, box
#对单个图片随机生成正常结节的crop区域

def crop_patch_2(img_size, crop_size, ratio, boundary_distance):
    w, h = crop_size["w"], crop_size["h"]
    w_ex_pixel = int((w * ratio)//2)
    h_ex_pixel = int((h * ratio)//2)

    y_0 = random.randint(boundary_distance + h_ex_pixel, img_size[0]-boundary_distance - h_ex_pixel)
    x_0 = random.randint(boundary_distance + w_ex_pixel, img_size[1]-boundary_distance - w_ex_pixel)   

    dim1_cut_min = max(0, y_0 - h_ex_pixel)
    dim1_cut_max = min(img_size[0] - 1, y_0 + h + h_ex_pixel)
    dim2_cut_min = max(0, x_0 - w_ex_pixel)
    dim2_cut_max = min(img_size[1] - 1, x_0 + w + w_ex_pixel)

    roi = [dim1_cut_min, dim1_cut_max, dim2_cut_min, dim2_cut_max]
    #return mask[dim1_cut_min:dim1_cut_max, dim2_cut_min:dim2_cut_max], roi    
    return roi

def get_random_box_between(inner_box, outer_box):
    x1, y1, w1, h1 = inner_box
    x2, y2, w2, h2 = outer_box

    try:
        random_x = random.randint(x2, x1)
        random_y = random.randint(y2, y1)
    except:
        random_x = x1
        random_y = y1
    try:
        end_x = random.randint(x1+w1, x2+w2)
        end_y = random.randint(y1+h1, y2+h2)
    except:
        end_x = x1+w1
        end_y = y1+h1
    random_w = end_x - random_x
    random_h = end_y - random_y
   
    random_box = [random_x, random_y, random_w, random_h]
    return random_box

def extend_by_max_edge(box, ratio=0.15, size=[]):
    x, y, w, h = box
    max_length = max(w, h)
    expand_pixel = int(max_length * ratio)

    dim1_cut_min = max(0, y - expand_pixel)
    dim1_cut_max = min(size[0] - 1, y + h + expand_pixel)
    dim2_cut_min = max(0, x - expand_pixel)
    dim2_cut_max = min(size[1] - 1, x + w + expand_pixel)
    
    new_box = [dim2_cut_min, dim1_cut_min, dim2_cut_max-dim2_cut_min, dim1_cut_max-dim1_cut_min]
    return new_box

def shrink_by_max_edge(box, ratio=0.1, size=[]):
    x, y, w, h = box
    max_length = max(w, h)
    
    shrink_pixel = int(max_length * ratio)
    
    # Calculate new bounding box coordinates
    new_x = x + shrink_pixel
    new_y = y + shrink_pixel
    new_w = w - 2 * shrink_pixel
    new_h = h - 2 * shrink_pixel

    # Ensure the new bounding box remains within the image boundaries
    new_x = max(new_x, 0)
    new_y = max(new_y, 0)
    new_w = min(new_w, size[1] - new_x)
    new_h = min(new_h, size[0] - new_y)
    
    new_box = [new_x, new_y, new_w, new_h]
    return new_box

def center_crop_new_box(box, img_size, patch_size=256):
    x, y, w, h = box
    x_center, y_center = x + w // 2, y + h // 2
    expand_pixel = patch_size// 2
    img_height, img_width = img_size

    x_left = max(x_center - expand_pixel, 0)
    y_top = max(y_center - expand_pixel, 0)
    x_right = min(x_center + (patch_size - expand_pixel), img_width)
    y_bottom = min(y_center + (patch_size - expand_pixel), img_height)

    return [int(x_left), int(y_top), int(x_right) - int(x_left), int(y_bottom) - int(y_top)]

