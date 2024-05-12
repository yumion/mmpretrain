import argparse
import collections
import functools
import operator

from mmengine.config import Config
from mmpretrain.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--phase",
        default="train",
        type=str,
        choices=["train", "test", "val"],
        help='phase of dataset to visualize, accept "train" "test" and "val".',
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.phase == "train":
        dataset_cfg = cfg.train_dataloader.dataset
    elif args.phase == "val":
        dataset_cfg = cfg.val_dataloader.dataset
    elif args.phase == "test":
        dataset_cfg = cfg.test_dataloader.dataset
    else:
        raise ValueError("'--phase' only support 'train', 'val' and 'test'.")

    dataset = build_dataset_wrapper(dataset_cfg)
    num_per_class = count_data_list(dataset)
    print(num_per_class)
    print(f"Total number of data: {sum(num_per_class.values())}")


def build_dataset_wrapper(dataset_cfg):
    dataset_type = dataset_cfg.get("type")
    if dataset_type == "ConcatDataset":
        datasets = dataset_cfg.get("datasets")
        return [build_dataset_wrapper(ds) for ds in datasets]
    elif dataset_type in ["RepeatDataset", "ClassBalancedDataset"]:
        dataset = build_dataset_wrapper(dataset_cfg.get("dataset"))
        return dataset
    else:
        return build_dataset(dataset_cfg)


def count_data_list(dataset):
    if isinstance(dataset, list):
        return dict(
            functools.reduce(
                operator.add, map(collections.Counter, [count_data_list(ds) for ds in dataset])
            )
        )

    classes = dataset.metainfo.get("classes")
    num_per_class = {c: 0 for c in classes}
    for i in range(len(dataset)):
        gt_label = dataset.get_data_info(i)["gt_label"]
        num_per_class[classes[gt_label]] += 1
    return num_per_class


if __name__ == "__main__":
    main()
