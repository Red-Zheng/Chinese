#!/bin/bash  
bashrc_CS="start......"  
echo "$bashrc_CS"

CONFIG_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/20241030_100304/vis_data/config.py
RESULT_PKL_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/result.pkl
OUTPUT_DIR=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/tools/analysis_tools/confusion_matrix.py "$CONFIG_PATH" "$RESULT_PKL_PATH" "$OUTPUT_DIR"

echo "$bashrc_CS"