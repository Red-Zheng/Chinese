auto_scale_lr = dict(base_batch_size=256, enable=False)
backend_args = None
data_root = 'datasets/handwritten_chinese_stroke_2021/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=5, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
launcher = 'pytorch'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 40
model = dict(
    backbone=dict(
        base_width=4,
        depth=101,
        frozen_stages=1,
        groups=64,
        dcn=dict(deform_groups=1, fallback_on_stride=False, type='DCNv2'),
        init_cfg=dict(
            checkpoint='open-mmlab://resnext101_64x4d', type='Pretrained'),
        norm_cfg=dict(requires_grad=True, type='BN'),
        norm_eval=True,
        num_stages=4,
        out_indices=(
            0,
            1,
            2,
            3,
        ),
        stage_with_dcn=(
            False,
            True,
            False,
            True,
            
        ),
        style='pytorch',
        type='ResNeXt'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_seg=True,
        pad_size_divisor=32,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            256,
            512,
            1024,
            2048,
        ],
        num_outs=5,
        out_channels=256,
        type='FPN'),
    
    rpn_head=dict(
        anchor_generator=dict(
            ratios=[
                0.5,
                1.0,
                2.0,
            ],
            scales=[
                8,
            ],
            strides=[
                4,
                8,
                16,
                32,
                64,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=256,
        in_channels=256,
        loss_bbox=dict(
            beta=0.1111111111111111, loss_weight=1.0, type='SmoothL1Loss'),
        loss_cls=dict(
            loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        type='RPNHead'),
    roi_head=dict(
        bbox_head=[
            dict(
            type='SABLHead',
            num_classes=25,
            cls_in_channels=256,
            reg_in_channels=256,
            roi_feat_size=7,
            reg_feat_up_ratio=2,
            reg_pre_kernel=3,
            reg_post_kernel=3,
            reg_pre_num=2,
            reg_post_num=1,
            cls_out_channels=1024,
            reg_offset_out_channels=256,
            reg_cls_out_channels=256,
            num_cls_fcs=1,
            num_reg_fcs=0,
            reg_class_agnostic=True,
            norm_cfg=None,
            bbox_coder=dict(
                type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.7),
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
            loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                               loss_weight=1.0)),
            dict(
                type='SABLHead',
                num_classes=25,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.5),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1,
                                loss_weight=1.0)),
            dict(
                type='SABLHead',
                num_classes=25,
                cls_in_channels=256,
                reg_in_channels=256,
                roi_feat_size=7,
                reg_feat_up_ratio=2,
                reg_pre_kernel=3,
                reg_post_kernel=3,
                reg_pre_num=2,
                reg_post_num=1,
                cls_out_channels=1024,
                reg_offset_out_channels=256,
                reg_cls_out_channels=256,
                num_cls_fcs=1,
                num_reg_fcs=0,
                reg_class_agnostic=True,
                norm_cfg=None,
                bbox_coder=dict(
                    type='BucketingBBoxCoder', num_buckets=14, scale_factor=1.3),
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
                loss_bbox_reg=dict(type='SmoothL1Loss', beta=0.1, loss_weight=1.0))
        ],
        bbox_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=7, sampling_ratio=0, type='RoIAlign'),
            type='GenericRoIExtractor'),
        interleaved=True,
        mask_head=[
            dict(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=dict(
                    loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
                num_classes=25,
                num_convs=4,
                type='HTCMaskHead',
                with_conv_res=False),
            dict(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=dict(
                    loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
                num_classes=25,
                num_convs=4,
                type='HTCMaskHead'),
            dict(
                conv_out_channels=256,
                in_channels=256,
                loss_mask=dict(
                    loss_weight=1.0, type='CrossEntropyLoss', use_mask=True),
                num_classes=25,
                num_convs=4,
                type='HTCMaskHead'),
        ],
        mask_info_flow=True,
        mask_roi_extractor=dict(
            featmap_strides=[
                4,
                8,
                16,
                32,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='GenericRoIExtractor'),
        num_stages=3,
        semantic_head=dict(
            conv_out_channels=256,
            fusion_level=1,
            in_channels=256,
            loss_seg=dict(
                ignore_index=255, loss_weight=0.5, type='CrossEntropyLoss'),
            num_classes=183,
            num_convs=4,
            num_ins=5,
            seg_scale_factor=0.125,
            type='FusedSemanticHead'),
        semantic_roi_extractor=dict(
            featmap_strides=[
                8,
            ],
            out_channels=256,
            roi_layer=dict(output_size=14, sampling_ratio=0, type='RoIAlign'),
            type='GenericRoIExtractor'),
        stage_loss_weights=[
            1,
            0.5,
            0.25,
        ],
        type='HybridTaskCascadeRoIHead'),
    test_cfg=dict(
        rcnn=dict(
            mask_thr_binary=0.5,
            max_per_img=100,
            nms=dict(iou_threshold=0.5, type='nms'),
            score_thr=0.001),
        rpn=dict(
            max_per_img=1000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=1000)),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    min_pos_iou=0.5,
                    neg_iou_thr=0.5,
                    pos_iou_thr=0.5,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    min_pos_iou=0.6,
                    neg_iou_thr=0.6,
                    pos_iou_thr=0.6,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
            dict(
                assigner=dict(
                    ignore_iof_thr=-1,
                    min_pos_iou=0.7,
                    neg_iou_thr=0.7,
                    pos_iou_thr=0.7,
                    type='MaxIoUAssigner'),
                debug=False,
                mask_size=28,
                pos_weight=-1,
                sampler=dict(
                    add_gt_as_proposals=True,
                    neg_pos_ub=-1,
                    num=512,
                    pos_fraction=0.25,
                    type='RandomSampler')),
        ],
        rpn=dict(
            allowed_border=0,
            assigner=dict(
                ignore_iof_thr=-1,
                min_pos_iou=0.3,
                neg_iou_thr=0.3,
                pos_iou_thr=0.7,
                type='MaxIoUAssigner'),
            debug=False,
            pos_weight=-1,
            sampler=dict(
                add_gt_as_proposals=False,
                neg_pos_ub=-1,
                num=256,
                pos_fraction=0.5,
                type='RandomSampler')),
        rpn_proposal=dict(
            max_per_img=2000,
            min_bbox_size=0,
            nms=dict(iou_threshold=0.7, type='nms'),
            nms_pre=2000)),
    type='HybridTaskCascade')
optim_wrapper = dict(
    optimizer=dict(lr=0.01, momentum=0.9, type='SGD', weight_decay=0.0001),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.0001,
        type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=40,
        gamma=0.1,
        milestones=[
            16,
            19,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2021.json',
        backend_args=None,
        data_prefix=dict(img='val2021/'),
        data_root='datasets/handwritten_chinese_stroke_2021/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640, 640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='datasets/handwritten_chinese_stroke_2021/annotations/instances_val2021.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640, 640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(max_epochs=25, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    batch_size=2,
    dataset=dict(
        ann_file='annotations/instances_train2021.json',
        backend_args=None,
        data_prefix=dict(
            img='/mnt/big_disk_0/gbw/new_mmdetection/mmdetection-main/datasets/handwritten_chinese_stroke_2021/train2021/',
            seg=
            '/mnt/big_disk_0/gbw/new_mmdetection/mmdetection-main/datasets/handwritten_chinese_stroke_2021/stuffthingmaps/train2021'
        ),
        data_root='datasets/handwritten_chinese_stroke_2021/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(
                type='LoadAnnotations',
                with_bbox=True,
                with_mask=True,
                with_seg=True),
            dict(keep_ratio=True, scale=(
                640, 640,
            ), type='Resize'),
            dict(prob=0.5, type='RandomFlip'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        type='LoadAnnotations', with_bbox=True, with_mask=True, with_seg=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    # 2. 几何变换增强（小角度+保持笔画结构）
    dict(type='RandomRotate', angle_range=(-15, 15), prob=0.5),  # 小角度旋转

    # 3. 颜色空间增强（严格控制参数范围）
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,       # ±32/255 (≈±12.5%)
        contrast_range=(0.8, 1.2), # 对比度±20%
        saturation_range=(0.9, 1.1), # 饱和度±10%
        hue_delta=9,              # 色相±5° (HSV范围0-180)
        p=0.7                     # 70%概率应用
    ),

    # 高斯噪声增强
    dict(
        type='Albu',
        transforms=[
            dict(
                type='GaussNoise',
                var_limit=(0, 15),  # 标准差范围0-15
                p=0.3              # 30%应用概率
            )
        ],
        bbox_params=None,
        keymap=None,
        update_pad_shape=False,
        skip_img_without_anno=True
    ),

    # dict(prob=0.5, type='RandomFlip'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        ann_file='annotations/instances_val2021.json',
        backend_args=None,
        data_prefix=dict(img='val2021/'),
        data_root='datasets/handwritten_chinese_stroke_2021/',
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640, 640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='datasets/handwritten_chinese_stroke_2021/annotations/instances_val2021.json',
    backend_args=None,
    format_only=False,
    metric=[
        'bbox',
        'segm',
    ],
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '/mnt/big_disk_0/gbw/new_mmdetection/mmdetection-main/test_work_dirs/htc_x101-64x4d-FPN'
