num_classes = 3
load_from = None

data_preprocessor = dict(
    num_classes=num_classes,
    # RGB format normalization parameters
    mean=[0, 0, 0],
    std=[255, 255, 255],
    # convert image from BGR to RGB
    to_rgb=True,
)
# Model settings
model = dict(
    data_preprocessor=data_preprocessor,
    type="TimmClassifier",
    model_name="maxvit_rmlp_base_rw_224.sw_in12k_ft_in1k",
    pretrained=True,
    num_classes=num_classes,
    loss=dict(
        type="LabelSmoothLoss",
        label_smooth_val=0.1,
    ),
    # backbone=dict(
    #     type="TIMMBackbone",
    #     model_name="maxvit_base_tf_512.in21k_ft_in1k",
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

# for batch in each gpu is 128, 8 gpu
# lr = 5e-4 * 128 * 8 / 512 = 0.001
optim_wrapper = dict(
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
