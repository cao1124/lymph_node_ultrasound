# thyroid_nodule_classify_randomcrop


# **本代码用于甲状腺结节分类**

1、randomcrop表示结节的图片在训练过程中在线随机crop（外扩比例随机）

2、main.py为入口文件，训练和测试模式通过main.py的_test()和_train()进行切换；

3、args.py中包含所有参数设置；

4、trainer目录下是主要的训练代码；

5、preprocess目录下为数据的预处理代码，包括：

    （1）about_crop.py包括crop相关的处理函数；

    （2）get_boxes.py为获取所有结节的外接矩形框的信息[x,y,w,h]，在训练前提前生成该文件，因为该过程比较耗时，所以提前提取好，训练时可以直接调用；

    （3）visualize_random_crop.py是随机crop的可视化例子；

    （4）split_dataset_image.py为按照病历划分数据集，数据列表是所有图片的名称，用于fixcrop的训练和验证、randomcrop的验证（随机crop只针对训练集）；

    （5）split_dataset_patient.py为按照病历划分数据集，数据列表是所有病历的名称，用于randomcrop的训练和验证、

6、refer目录下面包括训练和测试的图片的列表文件和label的序号对应文件。
