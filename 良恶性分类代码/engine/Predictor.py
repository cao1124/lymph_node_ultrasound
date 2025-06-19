import os
import warnings
warnings.filterwarnings('ignore')
import cv2
import pandas as pd
import torch
import pickle
import torch.nn as nn
import numpy as np
from _base import BaseEngine
from torch.utils.data import DataLoader
from sklearn import metrics
from tqdm import tqdm
from util import delong_auc_ci, bootstrap_auc_ci
from torchvision.transforms.functional import to_pil_image
from torchcam.methods import GradCAM
from torchcam.utils import overlay_mask
from preprocess.about_crop import resize_img_keep_ratio
from torchvision.transforms import transforms
from dataset import datasets_dict, HX_model_checkpoint_dict
from torchcam.methods import GradCAM

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


class Evaler(BaseEngine):

    def __init__(self,
                 cuda,
                 patch_size,
                 cls_model_path,
                 boxes_path=None,
                 domain_dir=None,
                 use_data=None):

        super().__init__(cuda)
        self.pool_size = (256, 448, 512, 600, 700, 800, 900, 1024)
        self.patch_size = patch_size
        self.cls_model = self.load_model(cls_model_path)
        self.__getinfo__(patch_size, boxes_path, domain_dir, use_data)

    def __getinfo__(self, patch_size, boxes_path, domain_dir, use_data):
        self.domain_dir = domain_dir
        self.boxes_path = boxes_path
        self.boxTune = None
        self.true_labels = []
        self.pred_labels = []
        self.data_loader = self.get_dataloader(use_data, 'test')


    def __call__(self):
        true_labels, pred_labels, _ = self.batch_predict()
        metrics_dict = self.compute_metrics(true_labels, pred_labels)
        return metrics_dict

    def compute_aucConfidence(self):
        true_labels, _, pred_probs = self.batch_predict()
        auc, ci_lower, ci_upper = bootstrap_auc_ci(true_labels, pred_probs)
        print(f'CI: [{ci_lower}, {ci_upper}]')

    def load_model(self, model_path):
        cls_model = torch.load(model_path, map_location='cpu')
        if isinstance(cls_model, nn.DataParallel):
            cls_model = cls_model.module
        cls_model = cls_model.to(self.cuda)
        # cls_model.eval()
        return cls_model

    def get_dataloader(self, dataset_name, dataset_type):
        UseDataset = datasets_dict[dataset_name]
        dataset = UseDataset(
            domain_dir=self.domain_dir,
            patch_size=self.patch_size,
            boxes_path=self.boxes_path,
            dataset_type=dataset_type,
            domain_list=None,
            crop_type=None,
            is_rgb=True,
            HX=True,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return dataloader

    def batch_predict(self):
        true_labels = []
        pred_labels = []
        positive_pred_probs = []
        for i, batch in enumerate(tqdm(self.data_loader, ncols=100)):
            patch_tensor, label, patch_img = batch[0], batch[1], batch[2]
            patch_tensor = patch_tensor.to(self.cuda)
            true_labels.extend(label)
            with torch.no_grad():
                output = self.cls_model(patch_tensor)
                pred = torch.softmax(output, dim=-1)
                pred_prob, pred_label = torch.max(pred, dim=-1)
                pred_labels.extend(pred_label.cpu().numpy().tolist())
                positive_pred_probs.extend(pred[:, 1].cpu().numpy().tolist())
        return true_labels, pred_labels, positive_pred_probs


    def single_predict(self, image, box):
        # 带提示的模型预测
        patch = self.click_add_prompt(image, box)
        patch_tensor = self.cls_preprocess(patch)
        output = self.cls_model(patch_tensor)
        pred = torch.softmax(output, dim=-1)
        pred_prob, pred_label = torch.max(pred, dim=-1)
        ans = self.gradCAM(patch_tensor, patch)
        return pred_label.cpu().detach().numpy()[0], pred_prob.cpu().detach().numpy()[0], ans



    def cls_preprocess(self, patch):
        patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        mean = [0.485, 0.456, 0.406]
        std = [0.228, 0.224, 0.225]
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        patch_tensor = transformation(patch).unsqueeze(0).to(self.cuda)
        return patch_tensor


    def gradCAM(self, patch_tensor, patch):
        with GradCAM(self.cls_model, target_layer='features') as cam_extractor:
            pred_tensor = self.cls_model(patch_tensor)
            activation_map = cam_extractor(pred_tensor.squeeze(0).argmax().item(), pred_tensor)
            activation_map = activation_map[0].cpu().detach().numpy().squeeze(0)
            activation_map = cv2.applyColorMap(np.uint8(255 * activation_map), cv2.COLORMAP_JET)
            activation_map = cv2.cvtColor(activation_map, cv2.COLOR_BGR2RGB)
            activation_map = cv2.resize(activation_map, (patch.shape[1], patch.shape[0]))
            csm = (1 - 0.6) * activation_map + 0.6 * patch
            csm = csm / np.max(csm)
            csm = np.uint8(255 * csm)
            return (csm, patch)


    def click_add_prompt(self, image, box):
        image_height, image_width = image.shape[:2]
        x, y, w, h = box
        window_size = binary_search_first_ge(self.pool_size, max(w, h))

        x_center, y_center = x + w // 2, y + h // 2
        expand_pixel = [window_size // 2, window_size // 2]
        x_left = max(x_center - expand_pixel[1], 0)
        y_top = max(y_center - expand_pixel[0], 0)
        x_right = min(x_center + (window_size - expand_pixel[1]), image_width)
        y_bottom = min(y_center + (window_size - expand_pixel[0]), image_height)

        prompt = {
            'x_loc': x - x_left,
            'y_loc': y - y_top,
            'w': w,
            'h': h,
        }

        patch = image[y_top:y_bottom, x_left:x_right]
        if patch.shape[:2] != (256, 256):
            for p in prompt:
                prompt[p] = int(prompt[p] * 256 / patch.shape[0])
            patch = cv2.resize(patch, (256, 256), interpolation=cv2.INTER_NEAREST)
        x_loc, y_loc, w, h = prompt['x_loc'], prompt['y_loc'], prompt['w'], prompt['h']
        cv2.rectangle(patch, (x_loc, y_loc), (x_loc + w, y_loc + h), (0, 255, 0), 1)
        return patch
    def compute_metrics(self, true_labels, pred_labels):

        metrics_dict = {}
        acc = metrics.accuracy_score(true_labels, pred_labels)
        precision = metrics.precision_score(true_labels, pred_labels, average='macro')
        recall = metrics.recall_score(true_labels, pred_labels, average='macro')
        auc = metrics.roc_auc_score(true_labels, pred_labels, average='macro')
        weighted_f1_score = metrics.f1_score(true_labels, pred_labels, average='weighted')

        y_true_benign = 0
        y_pred_benign = 0
        y_true_malig = 0
        y_pred_malig = 0

        for i in range(len(true_labels)):
            if true_labels[i] == 0 or true_labels[i] == '良性':
                y_true_benign += 1
                if pred_labels[i] == 0 or pred_labels[i] == '良性':
                    y_pred_benign += 1
            elif true_labels[i] == 1 or true_labels[i] == '恶性':
                y_true_malig += 1
                if pred_labels[i] == 1 or pred_labels[i] == '恶性':
                    y_pred_malig += 1
        sensitivity = y_pred_malig / y_true_malig if y_true_malig != 0 else 0
        specificity = y_pred_benign / y_true_benign if y_true_benign != 0 else 0


        metrics_dict.update(
            {
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'auc': auc,
                'weighted_f1': weighted_f1_score,
                'specificity': specificity,
                'sensitivity': sensitivity,
            }
        )
        return metrics_dict

if __name__ == '__main__':
    # prefix = '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院'
    # usedata_name = 'WINDOW_CROP_CENTER_EXPLICIT_PROMPT'
    # metrics_dict = {
    #         'model':[],
    #         'acc':[],
    #         'weighted_f1':[],
    #         'sensitivity':[],
    #         'specificity':[]}
    #
    # model_path_dict = {
    #     'KEEP_RATIO':
    #     {
    #     'convnext': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/convnext_large_train_id:1/2024_10_29_22_43_28/epo_59_weighted_f1_score_0.7105/model.pt',
    #     'densenet161': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/densenet161_train_id:1/2024_10_29_18_41_06/epo_40_weighted_f1_score_0.7161/model.pt',
    #     'efficientnet-b0': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_10_29_20_27_44/top_recall_model/epo_14_malign:0.7313_belign:0.7233/model.pt',
    #     'max_vit': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/max_vit_train_id:1/2024_10_30_14_02_06/epo_46_weighted_f1_score_0.6907/model.pt',
    #     'resnext101':'/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/resnext101_32x8d_train_id:1/2024_10_29_10_52_33/epo_47_weighted_f1_score_0.6955/model.pt',
    #     'swintransformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/swin_train_id:1/2024_10_30_12_11_18/epo_0_weighted_f1_score_0.604/model.pt',
    #     'vgg16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/vgg16_bn_train_id:1/2024_10_29_14_24_12/epo_65_weighted_f1_score_0.707/model.pt',
    #     'vison_transformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_KEEP_RATIO_batch:32_lr:0.0001_warmup:0.05/vit_b_16_train_id:1/2024_10_30_10_03_02/epo_69_weighted_f1_score_0.687/model.pt',
    #     },
    #     'RESIZE':
    #     {
    #         'convnext': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/convnext_large_train_id:1/2024_10_29_23_52_29/epo_99_weighted_f1_score_0.6965/model.pt',
    #         'densenet161': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/densenet161_train_id:1/2024_10_29_19_08_39/epo_68_weighted_f1_score_0.7082/model.pt',
    #         'efficientnet-b0': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_10_29_20_37_53/epo_58_weighted_f1_score_0.7426/model.pt',
    #         'max_vit': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/max_vit_train_id:1/2024_10_30_14_33_43/epo_82_weighted_f1_score_0.6958/model.pt',
    #         'resnext101': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/resnext101_32x8d_train_id:1/2024_10_29_11_36_12/epo_18_weighted_f1_score_0.7051/model.pt',
    #         'swintransformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/swin_train_id:1/2024_10_30_12_31_29/epo_99_weighted_f1_score_0.6448/model.pt',
    #         'vgg16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/vgg16_bn_train_id:1/2024_10_29_15_06_53/epo_51_weighted_f1_score_0.7224/model.pt',
    #         'vison_transformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_RESIZE_batch:32_lr:0.0001_warmup:0.05/vit_b_16_train_id:1/2024_10_30_10_36_10/epo_71_weighted_f1_score_0.6748/model.pt',
    #     },
    #     'WINDOW_CROP_CENTER':
    #     {
    #         'convnext': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/convnext_large_train_id:1/2024_10_29_20_55_09/epo_10_weighted_f1_score_0.6877/model.pt',
    #         'densenet161': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/densenet161_train_id:1/2024_10_29_17_27_47/epo_87_weighted_f1_score_0.7241/model.pt',
    #         'efficientnet-b0': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_10_29_19_45_42/top_recall_model/epo_34_malign:0.7242_belign:0.7087/model.pt',
    #         'max_vit': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/max_vit_train_id:1/2024_10_30_13_11_32/epo_83_weighted_f1_score_0.6945/model.pt',
    #         'resnext101': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/resnext101_32x8d_train_id:1/2024_10_28_17_34_47/epo_84_weighted_f1_score_0.7021/model.pt',
    #         'swintransformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/swin_train_id:1/2024_10_30_11_09_21/epo_7_weighted_f1_score_0.6301/model.pt',
    #         'vgg16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_10_29_19_45_42/epo_8_weighted_f1_score_0.7369/model.pt',
    #         'vison_transformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_batch:32_lr:0.0001_warmup:0.05/vit_b_16_train_id:1/2024_10_30_09_11_36/epo_31_weighted_f1_score_0.6587/model.pt',
    #     },
    #     'WINDOW_CROP_CENTER_EXPLICIT_PROMPT':
    #     {
    #         'convnext': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/convnext_large_train_id:1/2024_10_29_21_47_08/epo_16_weighted_f1_score_0.6787/model.pt',
    #         'densenet161': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/densenet161_train_id:1/2024_10_29_18_04_56/epo_74_weighted_f1_score_0.72/model.pt',
    #         'efficientnet-b0': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/efficientnet-b0_train_id:1/2024_10_29_20_14_12/epo_32_weighted_f1_score_0.7446/model.pt',
    #         'max_vit': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/max_vit_train_id:1/2024_10_30_13_43_43/epo_10_weighted_f1_score_0.6821/model.pt',
    #         'resnext101': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/resnext101_32x8d_train_id:1/2024_10_29_10_10_08/epo_39_weighted_f1_score_0.6869/model.pt',
    #         'swintransformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/swin_train_id:1/2024_10_30_11_32_11/epo_61_weighted_f1_score_0.6771/model.pt',
    #         'vgg16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/vgg16_bn_train_id:1/2024_10_29_13_44_37/epo_39_weighted_f1_score_0.7199/model.pt',
    #         'vison_transformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/淋巴结科研/上海中山医院/LymphNodClassification_WINDOW_CROP_CENTER_EXPLICIT_PROMPT_batch:32_lr:0.0001_warmup:0.05/vit_b_16_train_id:1/2024_10_30_09_38_35/epo_25_weighted_f1_score_0.6695/model.pt',
    #     }
    #
    # }
    # metrics_dict['model'] = list(model_path_dict[usedata_name].keys())
    # for model_name in model_path_dict[usedata_name]:
    #     checkpoint = model_path_dict[usedata_name][model_name]
    #     if 'vit' in model_name or 'transformer' in model_name:
    #         patch_size = 224
    #     else:
    #         patch_size = 256
    #
    #     cuda = 'cuda:0'
    #     boxes_path = '../refer/中山医院淋巴结数据_1028.pkl'
    #     domain_dir = '/mnt/data/hsy/数据/淋巴结检测数据/中山医院科研数据'
    #
    #     evaler = Evaler(
    #         cuda=cuda,
    #         patch_size=patch_size,
    #         boxes_path=boxes_path,
    #         cls_model_path=checkpoint,
    #         domain_dir=domain_dir,
    #         use_data=usedata_name,
    #     )
    #     test_metrics = evaler()
    #     for key in test_metrics:
    #         if key in metrics_dict:
    #             metrics_dict[key].append(round(test_metrics[key],4))
    #
    #
    # df = pd.DataFrame(metrics_dict)
    # df.to_csv(f'{usedata_name}_metrics.csv', index=False)

    for key in HX_model_checkpoint_dict:
        evaler = Evaler(
                cuda='cuda:0',
                patch_size=224,
                cls_model_path=HX_model_checkpoint_dict[key],
                domain_dir='',
                boxes_path='../refer/HX_lymph_data.pkl',
                use_data='KEEP_RATIO')


    # all_data =[]
    # save_dirs = '/mnt/data/hsy/experiment/华西彭老师淋巴结分类数据热力图'
    # with open('../refer/HX_lymph_data.pkl', 'rb') as fs:
    #     data = pickle.load(fs)
    #     for _t in ['test']:
    #         all_data.extend(data['HX'][_t])
    #
    # y_true = []
    # y_pred_prob = []
    # for sample in tqdm(all_data, total=len(all_data), desc='collecting patch label'):
    #     box = sample['boxes']
    #     image_path = sample['image_name']
    #     gt = sample['tags']
    #     int_box = list(map(int, box.split(',')))
    #     image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #     patch = evaler.click_add_prompt(image, int_box)
    #     pred_label, pred_prob, ans = evaler.single_predict(image, int_box)
    #
    #     y_pred_prob.append(pred_prob)
    #     y_true.append(1 if 'malign' in gt else 0)



        # pred = 'malign' if pred_label == 1 else 'belign'
        # csm, patch = ans
        # concat_res = np.concatenate([patch, csm], axis=1)
        #
        # save_path = os.path.join(save_dirs, f'真实标签_{gt}_预测标签_{pred}')
        # os.makedirs(save_path, exist_ok=True)
        # base_name = os.path.basename(image_path[:-4]) + '热图.png'
        # cv2.imwrite(os.path.join(save_path, base_name), concat_res)

        print(f'当前模型:{key}')
        evaler.compute_aucConfidence()

