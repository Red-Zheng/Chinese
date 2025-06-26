#!/bin/bash  
bashrc_CS="start......"  
echo "$bashrc_CS"
  
# 设置路径变量  
CONFIG_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_dcnv2/20241108_085640/vis_data/config.py
CHECKPOINT_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_dcnv2/epoch_25.pth
OUTPUT_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_dcnv2/result.pkl
OUTPUT_DIR=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_dcnv2
JSON_PATH=/mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN_GRoIE_SABL_dcnv2/20241108_085640/vis_data/scalars.json

# 执行测试脚本  
python tools/test.py "$CONFIG_PATH" "$CHECKPOINT_PATH" --out "$OUTPUT_PATH"

echo "$bashrc_CS"

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/tools/analysis_tools/confusion_matrix_1.py "$CONFIG_PATH" "$OUTPUT_PATH" "$OUTPUT_DIR"

echo "$bashrc_CS"

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/tools/analysis_tools/analyze_logs.py plot_curve "$JSON_PATH" --keys segm_mAP_50_95 segm_mAP_50 segm_mAP_75 --legend segm_mAP_50_95 segm_mAP_50 segm_mAP_75 --out "$OUTPUT_DIR/segm.png"

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/tools/analysis_tools/analyze_logs.py plot_curve "$JSON_PATH" --keys  bbox_mAP_50_95  bbox_mAP_50  bbox_mAP_75 --legend  bbox_mAP_50_95  bbox_mAP_50  bbox_mAP_75 --out "$OUTPUT_DIR/bbox.png"

echo "$bashrc_CS"

python /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/mybash/plot_loss.py "$JSON_PATH" "$OUTPUT_DIR/mean_loss_per_epoch.png"

echo "$bashrc_CS"