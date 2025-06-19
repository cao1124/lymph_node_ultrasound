from .PreprocessDatasets import ResizeDatasets
from .PreprocessDatasets import KeepRatioResizeDatasets
from .PreprocessDatasets import WindowCropCenterDatasets
from .PreprocessDatasets import WindowCropCenterExplictPrompt
from .PreprocessDatasets import NoShuffleKeepData

datasets_dict = \
    {
    'RESIZE': ResizeDatasets,
    'KEEP_RATIO': KeepRatioResizeDatasets,
    'WINDOW_CROP_CENTER': WindowCropCenterDatasets,
    'WINDOW_CROP_CENTER_EXPLICIT_PROMPT': WindowCropCenterExplictPrompt,
    'NoShuffleKeepRatio': NoShuffleKeepData,
    }

ZS_model_checkpoint_dict = \
    {
        'vit-b16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/vit_b_16_train_id:0/2025_01_09_19_35_35/epo_91_weighted_f1_score_0.8405/model.pt',
        'vgg16': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/vgg16_bn_train_id:0/2025_01_09_18_04_10/epo_75_weighted_f1_score_0.8738/model.pt',
        'swin-transformer': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/swin_train_id:0/2025_01_09_20_00_16/epo_88_weighted_f1_score_0.8629/model.pt',
        'resnet101': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/resnet101_train_id:0/2025_01_09_19_05_30/epo_47_weighted_f1_score_0.8631/model.pt',
        'resnet50': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/resnet50_train_id:0/2025_01_09_18_38_53/epo_72_weighted_f1_score_0.8517/model.pt',
        'max_vit': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/max_vit_train_id:0/2025_01_09_20_31_49/epo_32_weighted_f1_score_0.8861/model.pt',
        'eb0': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:0/2025_01_09_17_32_22/epo_10_weighted_f1_score_0.9084/model.pt',
        'densenet121': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_RESIZE_batch:64_lr:0.0001_warmup:0.1/densenet121_train_id:0/2025_01_09_19_22_58/epo_43_weighted_f1_score_0.8611/model.pt',
        'proposed': '/mnt/data/hsy/科研项目/良恶性分类代码/saved_weights/中山-良恶性筛选实验/中山医院/malignBenignClassification_WINDOW_CROP_CENTER_batch:64_lr:0.0001_warmup:0.1/efficientnet-b0_train_id:0/2025_01_09_17_26_55/epo_29_weighted_f1_score_0.9188/model.pt'
    }
