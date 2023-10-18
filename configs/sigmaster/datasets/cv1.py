data_root = "/mnt/cloudy_3/SigMA/engineering/atsushi/MA_TractionQuality/dataset/sigma_20230927_PD"
train = [2, 3, 4, 5]
val = 1

batch_size = 16

# dataset settings
dataset_type = "CustomDataset"
classes = ["A", "B", "C"]


train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Albu",
        transforms=[
            dict(type="RandomResizedCrop", height=224, width=224, interpolation=4),
            dict(type="HorizontalFlip", p=0.5),

        ],
    ),
    dict(type="PackInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="Albu",
        transforms=[
            dict(type="Resize", height=224, width=224, interpolation=4),
        ],
    ),
    dict(type="PackInputs"),
]

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type="ConcatDataset",
        datasets=[
            dict(
                type=dataset_type,
                data_root=data_root,
                ann_file=f"fold{i}_sigma_20230927_PD.txt",
                data_prefix="",
                with_label=True,
                classes=classes,
                pipeline=train_pipeline,
            )
            for i in train
        ],
    ),
    sampler=dict(type="DefaultSampler", shuffle=True),
)

val_dataloader = dict(
    batch_size=batch_size,
    num_workers=16,
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=f"fold{val}_sigma_20230927_PD.txt",
        data_prefix="",
        with_label=True,
        classes=classes,
        pipeline=test_pipeline,
    ),
    sampler=dict(type="DefaultSampler", shuffle=False),
)
val_evaluator = dict(type="SingleLabelMetric")

# If you want standard test, please manually configure the test dataset
test_dataloader = val_dataloader
test_evaluator = val_evaluator
