# runtime setting

# defaults to use registries in mmpretrain
default_scope = "mmpretrain"

# set visualizer
vis_backends = [
    dict(type="WandbVisBackend", init_kwargs=dict(project="SigMA", entity="yumion")),  # wandb
    dict(type="LocalVisBackend"),
]
visualizer = dict(
    type="UniversalVisualizer",
    vis_backends=vis_backends,
)

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type="IterTimerHook"),
    # print log every 100 iterations.
    logger=dict(type="LoggerHook", interval=100),
    # enable the parameter scheduler.
    param_scheduler=dict(type="ParamSchedulerHook"),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type="DistSamplerSeedHook"),
    # save checkpoint per epoch.
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=1,
        save_best="single-label/f1-score",
        greater_keys=["f1-score"],
    ),
    # validation results visualization, set True to enable it.
    visualization=dict(type="VisualizationHook", enable=True, interval=50),
)
# custom_hooks = [dict(type="EMAHook", momentum=1e-4, priority="ABOVE_NORMAL")]

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend="nccl"),
)


# set log level
log_level = "INFO"

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=3407, deterministic=False)
