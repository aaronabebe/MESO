import sys

# this is needed for fiftyone dataset import to work
sys.path.append("/home/aaron/dev/SSL_MFE")

import fiftyone as fo
from fiftyone import ViewField as F

DATASET_NAME = "SAILING_DATASET"
DATASET_DIR_16BIT = "/home/aaron/dev/data/data_16bit"
# DATASET_DIR = "/mnt/fiftyoneDB/Database/Annotation_Data/SAILING_DATASET"
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

SAILING_CLASSES_TOTAL = (
    'ALGAE', 'BIRD', 'BOAT', 'BOAT_WITHOUT_SAILS', 'BUOY', 'CONSTRUCTION', 'CONTAINER', 'CONTAINER_SHIP', 'CRUISE_SHIP',
    'DOLPHIN', 'FAR_AWAY_OBJECT', 'FISHING_BUOY', 'FISHING_SHIP', 'FLOTSAM', 'HARBOUR_BUOY', 'HORIZON', 'HUMAN',
    'HUMAN_IN_WATER', 'HUMAN_ON_BOARD', 'KAYAK', 'LEISURE_VEHICLE', 'MARITIME_VEHICLE', 'MOTORBOAT',
    'OBJECT_REFLECTION', 'SAILING_BOAT', 'SAILING_BOAT_WITH_CLOSED_SAILS', 'SAILING_BOAT_WITH_OPEN_SAILS', 'SEAGULL',
    'SHIP', 'SUN_REFLECTION', 'UNKNOWN', 'WATERTRACK'
)

SAILING_CLASSES_SUBSET_V1 = (
    "BOAT", "BOAT_WITHOUT_SAILS", "CONSTRUCTION", "CONTAINER_SHIP", "CRUISE_SHIP", "FAR_AWAY_OBJECT",
    "FISHING_SHIP", "HARBOUR_BUOY", "MARITIME_VEHICLE", "MOTORBOAT", "SAILING_BOAT", "SAILING_BOAT_WITH_CLOSED_SAILS",
    "SAILING_BOAT_WITH_OPEN_SAILS", "SHIP", "WATERTRACK"
)


def map_class_to_subset(label: str):
    label = next((k for k, v in SUBSET_CLASS_MAP.items() if label in v), None)
    if label is None:
        raise ValueError(f"Label {label} not found in class map")
    return label


def get_crop_size_filter(min_crop_size: int = 32):
    bbox_area = (
            F("$metadata.width")
            * F("bounding_box")[2]
            * F("$metadata.height")
            * F("bounding_box")[3]
    )
    return min_crop_size ** 2 < bbox_area


def get_class_filter():
    return F("label").is_in(SAILING_CLASSES_SUBSET_V1)


def get_dataset(
        dataset_dir: str,
        dataset_name: str = DATASET_NAME,
        dataset_dir_16bit: str = DATASET_DIR_16BIT, use_16bit: bool = False,
        ground_truth_label: str = GROUND_TRUTH_LABEL, min_crop_size: int = 1,
        split=(0.9, 0.1)
):
    # this needs to be set to load custom datatypes
    fo.config.module_path = "custom_embedded_files"

    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
    else:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir if not use_16bit else dataset_dir_16bit,
            dataset_type=fo.types.FiftyOneDataset,
            name=dataset_name,
            label_field=ground_truth_label
        )

    view = dataset.select_group_slices(['thermal_left', 'thermal_right'])

    # # set default filters for now
    filters = [get_crop_size_filter(min_crop_size), get_class_filter()]

    for f in filters:
        view = view.filter_labels(GROUND_TRUTH_LABEL, f)

    print('Imported', len(view), '16bit' if use_16bit else '8bit', 'samples after applying filters.')
    train_len, val_len = int(split[0] * len(view)), int(split[1] * len(view))
    return view.take(train_len), view.take(val_len)
