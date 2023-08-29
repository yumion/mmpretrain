import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("labels", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()

    data_prefix = args.dataset
    labels = read_txt(args.labels)
    annotation_path = args.labels.parent / f"{data_prefix.name}.txt"

    with annotation_path.open("w") as fw:
        for group_dir in data_prefix.glob("*"):
            # data_root/clip_000000
            for label_dir in group_dir.glob("*"):
                # data_root/clip_000000/class1
                label = label_dir.name
                label_index = labels.index(label)
                for input_path in label_dir.glob("*.png"):
                    # data_root/clip_000000/class1/frame_000000.png
                    input_file = str(input_path).replace(f"{data_prefix}/", "")
                    fw.write(f"{input_file} {label_index}\n")


def read_txt(path_txt):
    with path_txt.open() as fr:
        return [s.rstrip() for s in fr.readlines()]


if __name__ == "__main__":
    main()
