from datasets import DatasetDict, load_dataset

from shared import LOCAL_DATASET_PATH, MAX_SVG_LENGTH


SOURCE_DATASET_NAME = "xingxm/SVGX-Core-250k"


def is_short_svg(example):
    return len(example["svg_code"]) <= MAX_SVG_LENGTH


def main():
    ds = load_dataset(SOURCE_DATASET_NAME)
    filtered_splits = {
        split_name: split_ds[:5000].filter(is_short_svg)
        for split_name, split_ds in ds.items()
    }
    filtered_ds = DatasetDict(filtered_splits)
    filtered_ds.save_to_disk(LOCAL_DATASET_PATH)


if __name__ == "__main__":
    main()
