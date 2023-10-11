import argparse
from pathlib import Path

from sklearn import model_selection


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--suffix", type=str, default=".png")
    parser.add_argument(
        "--num-fold",
        "--num_fold",
        type=int,
        default=-1,
        help="divide folds for cross validation. -1 means doesn't divide.",
    )
    parser.add_argument(
        "--strategy",
        choices=["StratifiedGroupKFold", "GroupKFold", "StratifiedKFold", "KFold"],
        default="StratifiedGroupKFold",
    )
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7488)
    return parser.parse_args()


def main():
    args = parse_args()

    data_prefix = args.dataset
    labels = read_txt(args.labels)
    data = get_structure(data_prefix, labels, args.suffix)

    # foldに分けない
    if args.num_fold <= 0:
        annotation_path = args.labels.parent / f"{data_prefix.name}.txt"
        save_annotation(annotation_path, data["X"], data["y"])
        return

    # 分け方を指定
    kf = getattr(model_selection, args.strategy)(
        n_splits=args.num_fold, shuffle=args.shuffle, random_state=args.seed
    )

    for i, (_, test_idxes) in enumerate(kf.split(**data)):
        input_files = [data["X"][idx] for idx in test_idxes]
        ground_truths = [data["y"][idx] for idx in test_idxes]
        groups = [data["groups"][idx] for idx in test_idxes]
        print(f"Fold {i + 1}:")
        print(f"    index={test_idxes}")
        print(f"    group={groups}")
        print(f"    total={len(test_idxes)}")
        annotation_path = args.labels.parent / f"fold{i + 1}_{data_prefix.name}.txt"
        save_annotation(annotation_path, input_files, ground_truths)


def read_txt(path_txt):
    with path_txt.open() as fr:
        return [s.rstrip() for s in fr.readlines()]


def get_structure(data_root, labels, suffix):
    groups, inputs, ground_truths = [], [], []
    for group_dir in data_root.glob("*"):
        # data_root/clip_000000
        group_name = group_dir.name
        for label_dir in group_dir.glob("*"):
            # data_root/clip_000000/class1
            label = label_dir.name
            label_index = labels.index(label)
            for input_path in label_dir.glob(f"*{suffix}"):
                # data_root/clip_000000/class1/frame_000000.png
                input_file = str(input_path).replace(f"{data_root}/", "")

                groups.append(group_name)
                ground_truths.append(label_index)
                inputs.append(input_file)
    return {"X": inputs, "y": ground_truths, "groups": groups}


def save_annotation(annotation_path, inputs, ground_truths):
    with annotation_path.open("w") as fw:
        num_data = len(ground_truths)
        for idx in range(num_data):
            input_file = inputs[idx]
            label_index = ground_truths[idx]
            fw.write(f"{input_file} {label_index}\n")


if __name__ == "__main__":
    main()
