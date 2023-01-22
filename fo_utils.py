import fiftyone as fo
from fiftyone import ViewField as F

DATASET_NAME = "SAILING_DATASET"
DATASET_DIR = "/Users/aaronabebe/Downloads/20000_sample_aaron"
GROUND_TRUTH_LABEL = "ground_truth_det"


def get_dataset(
        dataset_name: str = DATASET_NAME, dataset_dir: str = DATASET_DIR,
        ground_truth_label: str = GROUND_TRUTH_LABEL, min_crop_size: int = 32,
        split=(0.8, 0.1)
):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name,
        label_field=ground_truth_label
    )
    bbox_area = (
            F("$metadata.width")
            * F("bounding_box")[2]
            * F("$metadata.height")
            * F("bounding_box")[3]
    )

    view = dataset.select_group_slices(['thermal_left', 'thermal_right'])
    view = view.filter_labels(
        "ground_truth_det", (min_crop_size ** 2 < bbox_area)
    )

    train_len, val_len = int(split[0] * len(view)), int(split[1] * len(view))
    return view.take(train_len), view.take(val_len)
