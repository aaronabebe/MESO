import fiftyone as fo
from fiftyone import ViewField as F

DATASET_NAME = "SAILING_DATASET"
DATASET_DIR = "/home/aaron/dev/data/20000_sample_aaron"
GROUND_TRUTH_LABEL = "ground_truth_det"

SUBSET_CLASS_MAP = {
    "BOAT": ["BOAT", "BOAT_WITHOUT_SAILS", "CONTAINER_SHIP", "CRUISE_SHIP", "FISHING_SHIP", "MOTORBOAT", "SAILING_BOAT",
             "SAILING_BOAT_WITH_CLOSED_SAILS", "SAILING_BOAT_WITH_OPEN_SAILS", "SHIP"],
    "HUMAN": ["HUMAN", "HUMAN_IN_WATER", "HUMAN_ON_BOARD"],
    "ANIMAL": ["BIRD", "DOLPHIN", "SEAGULL"],
    "BUOY": ["BUOY", "FISHING_BUOY", "HARBOUR_BUOY"],
    "CONSTRUCTION": ["CONSTRUCTION", "CONTAINER", "FAR_AWAY_OBJECT"],
    "VEHICLE": ["LEISURE_VEHICLE", "MARITIME_VEHICLE", "KAYAK"],
    "WATER_OBJECTS": ["ALGAE", "FLOTSAM", "HORIZON", "OBJECT_REFLECTION", "SUN_REFLECTION", "UNKNOWN", "WATERTRACK"]
}

SAILING_CLASSES_V1 = {
    "BOAT", "BOAT_WITHOUT_SAILS", "CONSTRUCTION", "CONTAINER_SHIP", "CRUISE_SHIP", "FAR_AWAY_OBJECT",
    "FISHING_SHIP", "HARBOUR_BUOY", "MARITIME_VEHICLE", "MOTORBOAT", "SAILING_BOAT", "SAILING_BOAT_WITH_CLOSED_SAILS",
    "SAILING_BOAT_WITH_OPEN_SAILS", "SHIP", "WATERTRACK"
}


def map_class_to_subset(label: str):
    label = next((k for k, v in SUBSET_CLASS_MAP.items() if label in v), None)
    if label is None:
        raise ValueError(f"Label {label} not found in class map")
    return label


def get_dataset(
        dataset_name: str = DATASET_NAME, dataset_dir: str = DATASET_DIR,
        ground_truth_label: str = GROUND_TRUTH_LABEL, min_crop_size: int = 32,
        split=(0.9, 0.1)
):
    # this needs to be set to load custom datatypes
    fo.config.module_path = "custom_embedded_files"

    try:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.FiftyOneDataset,
            name=dataset_name,
            label_field=ground_truth_label
        )
    except ValueError:
        dataset = fo.load_dataset(dataset_name)

    bbox_area = (
            F("$metadata.width")
            * F("bounding_box")[2]
            * F("$metadata.height")
            * F("bounding_box")[3]
    )

    view = dataset.select_group_slices(['thermal_left', 'thermal_right'])
    view = view.filter_labels(
        GROUND_TRUTH_LABEL, (min_crop_size ** 2 < bbox_area)
    )
    view = view.filter_labels(
        GROUND_TRUTH_LABEL, F("label").is_in(SAILING_CLASSES_V1)
    )
    train_len, val_len = int(split[0] * len(view)), int(split[1] * len(view))
    return view.take(train_len), view.take(val_len)
