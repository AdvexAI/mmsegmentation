_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/cityscapes.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]
crop_size = (800, 800)
image_size = (1024, 1024)
#stride = (170, 170)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.8,
            1.3,
        ),
        scale=image_size,
        type='RandomResize'),
    dict(cat_max_ratio=0.75, crop_size=crop_size, type='RandomCrop'),
    dict(prob=0.5, type='RandomFlip'),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet101_v1c', backbone=dict(depth=101),
    auxiliary_head=dict(loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=0.4)),
    decode_head=dict(loss_decode=dict(
            type='DiceLoss', use_sigmoid=False, loss_weight=1.0))
    #test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride)
)

optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0005)
