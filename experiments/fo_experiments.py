import fiftyone as fo
import  numpy as np
from pprint import pprint
from PIL import Image
import argparse
import torch
import torchvision
from fiftyone import ViewField as F

DATASET_NAME = "SAILING_DATASET"
DATASET_DIR = "/home/aaron/thesis/datasets/20000_sample_aaron"
GROUND_TRUTH_LABEL = "ground_truth_det"


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    def __init__(self, fo_dataset: fo.Dataset, transforms: torchvision.transforms = None,
                 ground_truth_label: str = None):
        self.samples = fo_dataset.match(
            F(f'{ground_truth_label}.detections') > 0)  # TODO change to get a list of all detections
        self.transforms = transforms
        self.img_paths = self.samples.values("filepath")
        self.ground_truth_label = ground_truth_label

        self.classes = self.samples.distinct(f'{self.ground_truth_label}.detections.label')
        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path)  # TODO handle 16bit and 8bit grayscale correctly while loading

        # crop detection bounding box from image
        imgs = []
        labels = []
        for detection in sample[self.ground_truth_label].detections:
            if detection.area > 12:
                detection = sample[self.ground_truth_label].detections[0]
                width = sample.metadata.width
                height = sample.metadata.height
                crop_box = (
                    (width * detection.bounding_box[0]) - (width * detection.bounding_box[2]),
                    (height * detection.bounding_box[1]) - (height * detection.bounding_box[3]),
                    (width * detection.bounding_box[0]) + (width * detection.bounding_box[2] * 2),
                    (height * detection.bounding_box[1]) + (height * detection.bounding_box[3] * 2)
                )
                crop = img.crop(crop_box)
                if self.transforms:
                    crop = self.transforms(crop)
                imgs.append(crop)
                label = self.labels_map_rev[detection.label]
                labels.append(label)

        return torch.from_numpy(np.array(imgs)), torch.as_tensor(labels)


def get_dataset(dataset_name: str = DATASET_NAME, dataset_dir: str = DATASET_DIR,
                ground_truth_label: str = GROUND_TRUTH_LABEL):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name,
        label_field=ground_truth_label
    )
    view = dataset.select_group_slices(['thermal_left', 'thermal_right'])
    print(view)

    return dataset, FiftyOneTorchDataset(view, ground_truth_label=GROUND_TRUTH_LABEL)


def get_classes_dict(dataset: fo.Dataset):
    # class_view = dataset.select_fields([GROUND_TRUTH_LABEL]).values(GROUND_TRUTH_LABEL)
    class_view = dataset.match(F(f"{GROUND_TRUTH_LABEL}.detections.label") > 0)

    class_count = {}

    for s in class_view:
        for d in s.ground_truth_det.detections:
            if d.label not in class_count:
                class_count[d.label] = 1
            else:
                class_count[d.label] += 1
    return class_count


def main():
    sample = 10
    dataset, torch_dataset = get_dataset()
    loader = torch.utils.data.DataLoader(torch_dataset, shuffle=True, num_workers=1)

    for i, (imgs, labels) in enumerate(loader):
        for img, label in zip(imgs, labels):
            img = Image.open(img.numpy())
            img.show()
            print(label)
        break




def server_main(dataset_name: str = DATASET_NAME, dataset_dir: str = DATASET_DIR,
                ground_truth_label: str = GROUND_TRUTH_LABEL):
    dataset = fo.Dataset.from_dir(
        dataset_dir=dataset_dir,
        dataset_type=fo.types.FiftyOneDataset,
        name=dataset_name,
        label_field=ground_truth_label
    )
    session = fo.launch_app(dataset)
    print("App launched")
    session.wait()

    return dataset


if __name__ == '__main__':
    # TODO set envs via code and not via shell
    # export PYTHONPATH="${PYTHONPATH}:/home/aaron/thesis/repos/SSL_MFE/"
    # export FIFTYONE_MODULE_PATH=custom_embedded_files
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch_server", action='store_true', default=False)
    args = parser.parse_args()
    if args.launch_server:
        server_main()
    else:
        main()
