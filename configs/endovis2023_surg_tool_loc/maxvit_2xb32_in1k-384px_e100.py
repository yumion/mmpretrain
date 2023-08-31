_base_ = [
    "../_base_/default_runtime.py",
]
randomness = dict(seed=3407, deterministic=False)
load_from = None

data_root = "/data2/shared/miccai/EndoVis2023/SurgToolLoc/classification/v1.1"
crop_size = 384
lr = 5e-4
max_epochs = 100
num_gpus = 2
batch_size = 16


# dataset settings
dataset_type = "CustomDataset"
classes = [
    "bipolar_dissector",
    "bipolar_forceps",
    "cadiere_forceps",
    "clip_applier",
    "force_bipolar",
    "grasping_retractor",
    "monopolar_curved_scissors",
    "needle_driver",
    "permanent_cautery_hook_spatula",
    "prograsp_forceps",
    "stapler",
    "suction_irrigator",
    "tip_up_fenestrated_grasper",
    "vessel_sealer",
    "other",
]
data_preprocessor = dict(
    num_classes=len(classes),
    # RGB format normalization parameters
    mean=[0.32858519781689005 * 255, 0.15265839395622285 * 255, 0.14655234887549404 * 255],
    std=[0.07691241763785549 * 255, 0.053818967599625046 * 255, 0.056615884572508365 * 255],
    # convert image from BGR to RGB
    to_rgb=True,
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Albu",
        transforms=[
            dict(
                type="PadIfNeeded",
                min_height=crop_size,
                min_width=crop_size,
                border_mode=0,
                value=0,
            ),
            dict(type="RandomRotate90", p=0.5),
            dict(type="RandomResizedCrop", height=crop_size, width=crop_size, interpolation=4),
            dict(type="HorizontalFlip", p=0.5),
            dict(type="VerticalFlip", p=0.5),
        ],
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Albu",
        transforms=[
            dict(
                type="PadIfNeeded",
                min_height=crop_size,
                min_width=crop_size,
                border_mode=0,
                value=0,
            ),
            dict(type="Resize", height=crop_size, width=crop_size, interpolation=4),
        ],
    ),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="meta/train.txt",
        data_prefix="train",
        with_label=True,
        classes=classes,
        pipeline=train_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file="meta/val.txt",
        data_prefix="val",
        with_label=True,
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="Accuracy", topk=(1,))

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator


# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.05, eps=1e-8, betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        norm_decay_mult=0.0,
        bias_decay_mult=0.0,
        flat_decay_mult=0.0,
        custom_keys={
            ".absolute_pos_embed": dict(decay_mult=0.0),
            ".relative_position_bias_table": dict(decay_mult=0.0),
        },
    ),
)

# learning policy
param_scheduler = [
    # warm up learning rate scheduler
    dict(
        type="LinearLR",
        start_factor=lr,
        by_epoch=True,
        end=max_epochs // 5,
        # update by iter
        convert_to_iter_based=True,
    ),
    # main learning rate scheduler
    dict(type="CosineAnnealingLR", eta_min=1e-5, by_epoch=True, begin=max_epochs // 5),
]

# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size=batch_size * num_gpus)


# runtime setting
# set visualizer
vis_backends = [dict(type="LocalVisBackend")]
visualizer = dict(type="UniversalVisualizer", vis_backends=vis_backends)
# configure default hooks
default_hooks = dict(
    # save checkpoint per epoch.
    checkpoint=dict(by_epoch=True, interval=1, max_keep_ckpts=1, save_best="accuracy/top1"),
    # validation results visualization, set True to enable it.
    visualization=dict(enable=True, interval=1000),
)
# custom_hooks = [dict(type="EMAHook", momentum=1e-4, priority="ABOVE_NORMAL")]


# Model settings
model = dict(
    data_preprocessor=data_preprocessor,
    type="TimmClassifier",
    model_name="maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k",
    pretrained=True,
    num_classes=len(classes),
    loss=dict(
        type="LabelSmoothLoss",
        label_smooth_val=0.1,
    ),
    # backbone=dict(
    #     type="TIMMBackbone",
    #     model_name="maxvit_rmlp_base_rw_384.sw_in12k_ft_in1k",
    #     features_only=True,
    #     pretrained=True,
    #     out_indices=-1,
    # ),
    # neck=dict(type="GlobalAveragePooling"),
    # head=dict(
    #     in_channels=768,  # (96, 192, 384, 768)
    #     num_classes=len(classes),
    #     loss=dict(
    #         type="LabelSmoothLoss",
    #         label_smooth_val=0.1,
    #     ),
    # ),
    init_cfg=dict(type="TruncNormal", layer=["Conv2d", "Linear"], std=0.02, bias=0.0),
    train_cfg=dict(
        augments=[
            dict(type="Mixup", alpha=0.8),
            dict(type="CutMix", alpha=1.0),
        ]
    ),
)
