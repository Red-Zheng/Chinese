#!/bin/bash  
bashrc_CS="start......"  
echo "$bashrc_CS"
  
# 设置路径变量  
CONFIG_PATH="/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/20241030_100304/vis_data/config.py"
CHECKPOINT_PATH="/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/epoch_25.pth"  
OUTPUT_PATH="/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/result.pkl"  
  
# 执行测试脚本  
python tools/test.py "$CONFIG_PATH" "$CHECKPOINT_PATH" --out "$OUTPUT_PATH"