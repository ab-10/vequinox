from datasets import Dataset, DatasetDict, load_dataset

from shared import LOCAL_DATASET_PATH, MAX_SVG_LENGTH


SOURCE_DATASET_NAME = "xingxm/SVGX-Core-250k"


def is_short_svg(example):
    return len(example["svg_code"]) <= MAX_SVG_LENGTH


def main():
    ds = load_dataset(SOURCE_DATASET_NAME)
    filtered_splits = {
        split_name: Dataset.from_dict(split_ds)
        if isinstance(split_ds, dict)
        else split_ds
        for split_name, split_ds in ds.items()
    }
    limited_and_filtered_splits = {
        split_name: split_ds.select(range(min(5000, len(split_ds)))).filter(
            is_short_svg
        )
        for split_name, split_ds in filtered_splits.items()
    }
    filtered_ds = DatasetDict(limited_and_filtered_splits)
    filtered_ds.save_to_disk(LOCAL_DATASET_PATH)


if __name__ == "__main__":
    main()
