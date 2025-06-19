#set -e
#echo '中山测试'
#model_names=('vit-b16'
#            'swin-transformer'
#            'max_vit'
#            'resnet50'
#            'resnet101'
#            'eb0'
#            'densenet121'
#            'vgg16')
#
#
#for idx in "${!model_names[@]}"; do
#  test_model_name="${model_names[$idx]}"
#  index=$((idx+1))
#  echo "验证第:$index 个模型: $test_model_name"
#  # 正确的拼接方式
#  # log_dir="logs/lymphatic_train_doctordata_test_physiologicaldata_binary_cls_${index}"
#  # 运行命令
#  python main.py --model_name $test_model_name  --device_id '1' --dataset_name 'RESIZE'
#done

python main.py --model_name 'proposed'  --device_id '1' --dataset_name 'WINDOW_CROP_CENTER'