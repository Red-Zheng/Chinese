python tools/analysis_tools/analyze_logs.py plot_curve /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/20241030_100304/vis_data/scalars.json --keys segm_mAP_50_95 segm_mAP_50 segm_mAP_75 --legend segm_mAP_50_95 segm_mAP_50 segm_mAP_75 --out /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/segm.png

python tools/analysis_tools/analyze_logs.py plot_curve /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/20241030_100304/vis_data/scalars.json --keys  bbox_mAP_50_95  bbox_mAP_50  bbox_mAP_75 --legend  bbox_mAP_50_95  bbox_mAP_50  bbox_mAP_75 --out /mnt/big_disk/gbw/new_mmdetection/mmdetection-main/work_dirs/htc_x101-64x4d-FPN/bbox.png

