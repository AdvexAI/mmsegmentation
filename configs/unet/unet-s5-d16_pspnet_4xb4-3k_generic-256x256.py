_base_ = [
    '../_base_/models/pspnet_unet_s5-d16.py', '../_base_/datasets/cityscapes.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (256, 256)
image_size = (1024, 1024)
stride = (170, 170)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        keep_ratio=True,
        ratio_range=(
            0.5,
            2.0,
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
    test_cfg=dict(crop_size=crop_size, stride=stride))
