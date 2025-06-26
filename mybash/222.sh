#!/bin/bash  
bashrc_CS="start......"  
echo "$bashrc_CS"
  
# 设置路径变量  
CONFIG_PATH=/mnt/big_disk/gbw/strokeExtract_mmdet_2024/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_11_4/20241105_085119/vis_data/config.py
CHECKPOINT_PATH=/mnt/big_disk/gbw/strokeExtract_mmdet_2024/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_11_4/epoch_25.pth
OUTPUT_PATH=/mnt/big_disk/gbw/strokeExtract_mmdet_2024/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_11_4/result1.pkl
OUTPUT_DIR=/mnt/big_disk/gbw/strokeExtract_mmdet_2024/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_11_4

# 执行测试脚本  
python tools/test.py "$CONFIG_PATH" "$CHECKPOINT_PATH" --out "$OUTPUT_PATH"

echo "$bashrc_CS"

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/tools/analysis_tools/confusion_matrix_2.py "$CONFIG_PATH" "$OUTPUT_PATH" "$OUTPUT_DIR"
