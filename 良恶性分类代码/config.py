class Config_LymphNod_Classification:
    running_mode = 'test'
    # choice: ['train', 'test']
    labels_path = 'refer/label.txt'
    num_classes = 2
    train_batch_size = 64
    eval_batch_size = 64
    epochs = 100
    early_stop_epoch = 50
    # Optimizer settings
    lr = 1e-4
    lr_warmup = 0.005

    #file settings
    domain_dir = '/mnt/data/hsy/科研项目/良恶性分类代码/filter_data/上海中山'
    patch_boxes_path = 'refer/中山医院淋巴结数据_一致性筛选.pkl'


    # save settings
    CLS = '中山-淋巴瘤转移性分类实验'
    # CLS = 'NNNetWork有效性验证-训练测试同时添加噪声'
    hospital_name = '中山医院'
    meta_dataset = 'malignBenignClassification'
    # recover training
    resume_train = False
    resume_dir = ''

def getconfig(task):
    if task == 'lymph_nod_cls':
        config = Config_LymphNod_Classification
    else:
        raise ValueError(f"task: {task} is not supported")
    return config
