_base_ = [
    "datasets/cv1_1018.py",
    "models/maxvit_base.py",
    "runtime.py",
]

lr = 5e-4
eta_min_lr = 1e-5
max_epochs = 100
num_gpus = 1

# learning policy
optim_wrapper = dict(
    optimizer=dict(
        type="AdamW",
        lr=lr,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999),
    )
)

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
    dict(type="CosineAnnealingLR", eta_min=eta_min_lr, by_epoch=True, begin=max_epochs // 5),
]


# train, val, test setting
train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict()
test_cfg = dict()

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# based on the actual training batch size.
auto_scale_lr = dict(base_batch_size={{_base_.batch_size}} * num_gpus)
