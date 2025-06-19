import csv
import cv2
import sys
import json
import copy
import os
import re
import random
import numpy as np
import torch
import time
import json
import logging
from tqdm import tqdm
from sklearn import metrics
from typing import List
from sklearn.metrics import roc_auc_score
from scipy.stats import norm
import six
import torch.nn.functional as F
logger=logging.getLogger('main_log.util')

CSV_DELIMETER = ';'


def read_image(path, is_rgb=True, bgr2rgb=False):
    if is_rgb:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr2rgb:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return image

def mask2box(mask_path, type='xywh'):

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    all_contours = []
    for contour in contours:
        all_contours.append(contour)
        box = cv2.boundingRect(contour)
        if type == 'xywh':
            boxes.append(box)
        elif type == 'xyxy':
            x, y, w, h = box
            boxes.append([x, y, x+w, y+h])
    return boxes, all_contours



def set_stream_loger():
    logger = logging.getLogger('/.log')
    logger.setLevel(level=logging.INFO)
    ch = logging.StreamHandler()
    Formarter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
    ch.setFormatter(Formarter)
    logger.addHandler(ch)
    return logger

def create_directories_file(f):
    d = os.path.dirname(f)

    if d and not os.path.exists(d):
        os.makedirs(d)

    return f


def create_directories_dir(d):
    if d and not os.path.exists(d):
        os.makedirs(d)
    return d


def read_csv(file_path):
    lines = []
    with open(file_path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=CSV_DELIMETER, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            lines.append(row)
    return lines[0], lines[1:]


def save_dict(log_path, dic, name):
    # save arguments
    # 1. as json
    path = os.path.join(log_path, '%s.json' % name)
    f = open(path, 'w')
    json.dump(vars(dic), f)
    f.close()

    # 2. as string
    path = os.path.join(log_path, '%s.txt' % name)
    f = open(path, 'w')
    args_str = ["%s = %s" % (key, value) for key, value in vars(dic).items()]
    f.write('\n'.join(args_str))
    f.close()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reset_logger(logger):
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    for f in logger.filters[:]:
        logger.removeFilters(f)


def to_device(batch, device):
    converted_batch = dict()
    for key in batch.keys():
        if torch.is_tensor(batch[key]):
            converted_batch[key] = batch[key].to(device)
        elif type(batch[key]) is np.ndarray:
            converted_batch[key] = torch.from_numpy(batch[key]).to(device)
        else:
            converted_batch[key] = batch[key]
    return converted_batch


def init_logger(log_dir="./log"):
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    run_key = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(time.time()))
    log_file = os.path.join(log_dir,run_key+'.log')

    logger = logging.getLogger('main_log')
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler(filename=log_file)
    fh.setLevel(logging.DEBUG)
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('[%(asctime)s] %(filename)s->%(funcName)s line:%(lineno)d [%(levelname)s]%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info("Logger Initialized.")


def save_json_by_line(result, des_path):
    #*将预测结果保存为json文件到指定路径 
    with open(des_path, 'w', encoding='utf-8') as fw:
        for obj in tqdm(result, desc=f'saving data to {des_path}'):
            line = json.dumps(obj, ensure_ascii=False) + '\n'
            fw.write(line)
            fw.flush()


def read_text(path):
    """[读文本的内容]

    Args:
        path ([str]): [文本路径]

    Returns:
        res [str]]: [文本内容]
    """
    with open(path,'r') as fs:
        #res = fs.read().split('\n')[0]
        res = fs.read().replace('\n',' ')
    return res


def swap(v1, v2):
    return v2, v1


def write_csv(data,path):
    with open(path, 'wt') as fs:
        tsv_writer = csv.writer(fs, delimiter='\t')  
        for i in data:
            tsv_writer.writerow(i)


def read_csv(input_file, quotechar=None):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)
        return lines

# compute_metrics(np.array([12,23,45]),np.array(([12,23,12]),average='macro')
def compute_metrics(y_true, y_pred, average='macro', decimal_place=4):
    assert len(y_true) == len(y_pred)
    precision = metrics.precision_score(y_true, y_pred, average=average)
    recall = metrics.recall_score(y_true, y_pred, average=average)
    f1_score = metrics.f1_score(y_true, y_pred, average=average)
    weighted_f1_score = metrics.f1_score(y_true, y_pred, average='weighted')
    acc = metrics.accuracy_score(y_true, y_pred)
    kappa = metrics.cohen_kappa_score(y_true, y_pred)

    y_true_benign = 0
    y_pred_benign = 0
    y_true_malig = 0
    y_pred_malig = 0

    for i in range(len(y_true)):
        if y_true[i] == 0 or y_true[i] == '良性':
            y_true_benign += 1
            if y_pred[i] == 0 or y_pred[i] == '良性':
                y_pred_benign += 1
        elif y_true[i] == 1 or y_true[i] == '恶性':
            y_true_malig += 1
            if y_pred[i] == 1 or y_pred[i] == '恶性':
                y_pred_malig += 1
    sensitivity = y_pred_malig / y_true_malig if y_true_malig!=0 else 0
    specificity = y_pred_benign / y_true_benign if y_true_benign!=0 else 0
    avg_acc = (sensitivity + specificity)/2
    benign_malig_acc = (y_pred_benign + y_pred_malig) / (y_true_benign + y_true_malig + 1e-7)

    return {
        "acc": round(acc,decimal_place),
        "precision": round(precision,decimal_place),
        "recall": round(recall,decimal_place),
        "macro_f1_score": round(f1_score,decimal_place),
        "sensitivity": round(sensitivity, decimal_place),
        "specificity": round(specificity, decimal_place),
        "weighted_f1_score": round(weighted_f1_score, decimal_place),
        'benign_malig_acc' : round(benign_malig_acc, decimal_place),
        'kappa' : round(kappa, decimal_place)
    }


def read_json_by_line(file_path):
    data=[]
    count=0
    with open(file_path,'r',encoding='utf8') as f:  
        for line in f:
            try:   
                data.append(json.loads(line))
            except:
                logger.error(f'第{count}行读取错误！！！')
                continue             
            count+=1
    logger.info(f'正确读取的原始文本数量为{len(data)}')
    return data


def read_json(file_path):
    data=[]
    with open(file_path,'r', encoding='utf-8')as fp:
        data = json.load(fp)
    # logger.info(f'正确读取的原始文本数量为{len(data)}')
    return data


def save_txt(data,path):
    with open(path,'w')as fs:
        for i in data:
            fs.write(i+'\n')


def calculate_metrics(confusion_matrix):
    TP = confusion_matrix[1][1]
    TN = confusion_matrix[0][0]
    FP = confusion_matrix[0][1]
    FN = confusion_matrix[1][0]

    specificity = TN / (TN + FP)
    sensitivity = TP / (TP + FN)
    precision_positive = TP / (TP + FP)
    precision_negative = TN / (TN + FN)
    precision = TP / (TP + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    weighted_f1 = 2 * precision * sensitivity / (precision + sensitivity)

    return {
        "Specificity": specificity,
        "Sensitivity": sensitivity,
        "Precision Positive": precision_positive,
        "Precision Negative": precision_negative,
        "Precision": precision,
        "Accuracy": accuracy,
        "Weighted F1": weighted_f1
    }


def binary_search_first_ge(arr, target):
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] >= target:
            result = mid
            right = mid - 1
        else:
            left = mid + 1

    return arr[result]

def find_num(str:str):
    num = re.findall(r'\d+', str)
    num = list(map(int, num))
    num = np.array(num).reshape(-1, 2)
    return num

def json2mm(json_path:str,
            img_path:str,
            part_keys:List,):

    json_data = read_json(json_path)
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    image_height, image_width = image.shape[:2]
    mask = np.zeros(( image_height, image_width), dtype=np.uint8)
    mark = copy.deepcopy(image)
    organ, perContour, partContour = part_keys

    contour_info = json_data['parts'][organ][perContour][partContour]
    for contour in contour_info:
        contour_str = contour['contour']
        contour_array = find_num(contour_str)
        cv2.drawContours(mask, [contour_array], -1, 255, -1)
        cv2.drawContours(mark, [contour_array], -1, (0, 255, 0), 1)
    return mask, mark
def delong_roc_variance(y_true, y_pred):
    """
    Calculate the variance of AUC using DeLong's method.

    Args:
        y_true: Array of true binary labels (0 or 1).
        y_pred: Array of predicted probabilities.

    Returns:
        auc: AUC value.
        auc_variance: Variance of the AUC.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    positive_scores = y_pred[y_true == 1]
    negative_scores = y_pred[y_true == 0]

    n_pos = len(positive_scores)
    n_neg = len(negative_scores)

    # Rank the scores
    scores = np.concatenate([positive_scores, negative_scores])
    rank = np.argsort(np.argsort(scores))

    # Rank sums for positive and negative
    positive_ranks = rank[:n_pos]
    negative_ranks = rank[n_pos:]

    auc = (np.sum(positive_ranks) - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)

    # Variance calculation
    v01 = np.var(positive_ranks, ddof=1)
    v10 = np.var(negative_ranks, ddof=1)

    auc_variance = (v01 / n_pos + v10 / n_neg) / (n_pos + n_neg - 1)

    return auc, auc_variance

def delong_auc_ci(y_true, y_pred, confidence_level=0.95):
    """
    Calculate AUC and its confidence interval using DeLong's method.

    Args:
        y_true: Array of true binary labels (0 or 1).
        y_pred: Array of predicted probabilities.
        confidence_level: Confidence level (e.g., 0.95 for 95% CI).

    Returns:
        auc: Calculated AUC value.
        ci_lower: Lower bound of the confidence interval.
        ci_upper: Upper bound of the confidence interval.
    """
    auc, auc_variance = delong_roc_variance(y_true, y_pred)
    alpha = 1 - confidence_level
    z = norm.ppf(1 - alpha / 2)
    ci_lower = auc - z * np.sqrt(auc_variance)
    ci_upper = auc + z * np.sqrt(auc_variance)

    return auc, ci_lower, ci_upper


def bootstrap_auc_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    """
    Calculate AUC and its confidence interval using bootstrap resampling.

    Args:
        y_true (array-like): Ground truth binary labels (0 or 1).
        y_pred (array-like): Predicted probabilities for the positive class.
        n_bootstrap (int): Number of bootstrap samples. Default is 1000.
        alpha (float): Confidence level. Default is 0.95 (95% CI).

    Returns:
        auc_mean (float): Mean AUC from bootstrap samples.
        ci (tuple): Confidence interval (lower, upper).
    """
    try:
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    except:
        raise TypeError

    # Store bootstrap AUC scores
    bootstrapped_aucs = []
    n_samples = len(y_true)

    # Perform bootstrap resampling
    for _ in range(n_bootstrap):
        # Randomly sample with replacement
        indices = np.random.choice(range(n_samples), size=n_samples, replace=True)
        if len(np.unique(y_true[indices])) < 2:  # Avoid invalid AUC computation
            continue
        auc = roc_auc_score(y_true[indices], y_pred[indices])
        bootstrapped_aucs.append(auc)

    # Sort bootstrap AUCs to calculate percentiles
    bootstrapped_aucs = np.array(bootstrapped_aucs)
    lower_bound = np.percentile(bootstrapped_aucs, (1 - alpha) / 2 * 100)
    upper_bound = np.percentile(bootstrapped_aucs, (1 + alpha) / 2 * 100)

    # Mean AUC and confidence interval
    auc_mean = np.mean(bootstrapped_aucs)
    return auc_mean, lower_bound, upper_bound
