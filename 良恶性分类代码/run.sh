#!/bin/bash
set -e
#echo '淋巴瘤vs恶性-数据交叉验证'
#datasets_name=('KEEP_RATIO')
#models=('efficientnet-b0')
#fold_nums=('fold_1'
#          'fold_2'
#          'fold_3'
#          'fold_4'
#          'fold_5')
#for jdx in "${!models[@]}"; do
#  for idx in "${!datasets_name[@]}"; do
#    for fdx in "${!fold_nums[@]}"; do
#      model_name="${models[$jdx]}"
#      data_name="${datasets_name[$idx]}"
#      fold_num="${fold_nums[$fdx]}"
#      echo "Traing with model: $model_name and data: $data_name"
#      python main.py --device_id '3' --seed 1000 --dataset_name $data_name --train_id $fdx --model_name $model_name --criterion_name 'Weighted_CrossEntropy' --fold_num $fold_num
#    done
#  done
#done

echo 'RESIZED'
models=('efficientnet-b0'
        'vgg16_bn'
        'resnet50'
        'resnet101'
        'densenet121'
        'vit_b_16'
        'swin'
        'max_vit')
for idx in "${!models[@]}"; do
  model_name="${models[$idx]}"
  python main.py --device_id '7' --seed 1000 --dataset_name 'RESIZED' --train_id 1 --model_name $model_name --criterion_name 'Weighted_CrossEntropy'
done


#echo '华西彭老师科研代码训练'
#datasets_name=('KEEP_RATIO')
#models=('efficientnet-b0'
#        'vgg16_bn'
#        'resnet50'
#        'resnet101'
#        'densenet121'
#        'vit_b_16'
#        'swin'
#        'max_vit'
#        'resnet50_pretrain')
#
#for jdx in "${!models[@]}"; do
#  for idx in "${!datasets_name[@]}"; do
#    model_name="${models[$jdx]}"
#    data_name="${datasets_name[$idx]}"
#    echo "Traing with model: $model_name and data: $data_name"
#    python main.py --device_id '3' --seed 1000 --dataset_name $data_name --train_id 0 --model_name $model_name --criterion_name 'CrossEntropy'
#    done
#done