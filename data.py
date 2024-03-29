import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import RandomSampler, Subset

from fo_utils import GROUND_TRUTH_LABEL, map_class_to_subset
from transforms import default_mnist_transforms, default_fifty_one_transforms, default_cifar10_transforms, \
    default_fashion_mnist_transforms
from utils import CIFAR_10_CORRUPTIONS, DEFAULT_DATA_DIR


def get_dataloader(name: str, subset: int, transforms: torchvision.transforms = None, train: bool = True,
                   num_workers: int = 0, **kwargs) -> torch.utils.data.DataLoader:
    """
    Returns the dataloader for a given dataset.
    :return:
    """
    if name != 'fiftyone' and 'fo_dataset' in kwargs:
        del kwargs['fo_dataset']

    if name == 'cifar10':
        return _get_cifar10(train, transforms, num_workers, subset, **kwargs)
    elif name == 'cifar10-c':
        return _get_cifar10c(transforms, num_workers, subset, **kwargs)
    elif name == 'mnist':
        return _get_mnist(train, transforms, num_workers, subset, **kwargs)
    elif name == 'fashion-mnist':
        return _get_fashion_mnist(train, transforms, num_workers, subset, **kwargs)
    elif name == 'fiftyone':
        return _get_fifty_one(transforms, num_workers, subset, **kwargs)
    elif name == 'fiftyone-16':
        return _get_fifty_one(transforms, num_workers, subset, **kwargs)
    elif name == 'massmind':
        return _get_massmind(transforms, num_workers, subset, **kwargs)
    raise NotImplementedError(f'No such dataloader: {name}')


def _get_fashion_mnist(train: bool, transforms: torchvision.transforms, num_workers: int, subset: int,
                       **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.FashionMNIST(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_fashion_mnist_transforms(),
    )
    if subset > 0:
        trainset = Subset(trainset, range(0, subset))
    return torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, **kwargs
    )


def _get_mnist(train: bool, transforms: torchvision.transforms, num_workers: int, subset: int,
               **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.MNIST(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_mnist_transforms(),
    )
    if subset > 0:
        trainset = Subset(trainset, range(0, subset))
    return torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, **kwargs
    )


def _get_fifty_one(transforms: torchvision.transforms, num_workers: int, subset: int, fo_dataset=None,
                   **kwargs) -> torch.utils.data.DataLoader:
    # load fifty one dataset
    trainset = SailingCropDataset(
        fo_dataset=fo_dataset,
        transform=transforms or default_fifty_one_transforms(),
    )
    if subset > 0:
        trainset = Subset(trainset, range(0, subset))
    return torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, **kwargs
    )


def _get_cifar10c(transforms: torchvision.transforms, num_workers: int, subset: int,
                  cname: str = random.choice(CIFAR_10_CORRUPTIONS), **kwargs):
    evalset = CIFAR10CDataset(
        './data/CIFAR-10-C',
        cname,
        tranform=transforms or default_cifar10_transforms()
    )
    if subset > 0:
        evalset = Subset(evalset, range(0, subset))
    return torch.utils.data.DataLoader(
        evalset, shuffle=False, num_workers=num_workers, **kwargs
    )


def _get_cifar10(train: bool, transforms: torchvision.transforms, num_workers: int, subset: int,
                 **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_cifar10_transforms(),
    )
    if subset > 0:
        trainset = Subset(trainset, range(0, subset))
    return torch.utils.data.DataLoader(
        trainset, shuffle=True, num_workers=num_workers, **kwargs
    )


def _get_massmind(transforms: torchvision.transforms, num_workers: int, subset: int, **kwargs):
    # TODO implement this properly
    raise NotImplementedError


class CIFAR10CDataset(torchvision.datasets.VisionDataset):
    def __init__(self, root: str, name: str, tranform=None, target_transform=None):
        super(CIFAR10CDataset, self).__init__(root, transform=tranform, target_transform=target_transform)
        data_path = os.path.join(root, f'{name}.npy')
        target_path = os.path.join(root, 'labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(target_path)

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


class SailingCropDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            fo_dataset,
            transform: torchvision.transforms = None,
            ground_truth_label: str = GROUND_TRUTH_LABEL,
            use_16bit: bool = False,
            use_class_subset: bool = False,
    ):
        self.samples = fo_dataset
        self.sample_map = {}

        running_idx = 0
        for i, sample in enumerate(self.samples):
            for j in range(len(sample[ground_truth_label].detections)):
                self.sample_map[running_idx] = (i, j)
                running_idx += 1

        self.transforms = transform
        self.img_paths = self.samples.values("filepath")
        self.ground_truth_label = ground_truth_label
        self.use_class_subset = use_class_subset
        self.use_16bit = use_16bit

        all_classes = self.samples.distinct(f'{self.ground_truth_label}.detections.label')
        if self.use_class_subset:
            self.classes = {map_class_to_subset(c) for c in all_classes}
        else:
            self.classes = all_classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return sum(len(sample[self.ground_truth_label].detections) for sample in self.samples)

    def __getitem__(self, idx):
        img_path = self.img_paths[self.sample_map[idx][0]]
        sample = self.samples[img_path]

        if self.use_16bit:
            img = Image.open(img_path).convert("I")
        else:
            img = Image.open(img_path).convert("L")

        detection = sample[self.ground_truth_label].detections[self.sample_map[idx][1]]
        width = sample.metadata.width
        height = sample.metadata.height
        crop_box = (
            (width * detection.bounding_box[0]),
            (height * detection.bounding_box[1]),
            (width * detection.bounding_box[0]) + (width * detection.bounding_box[2]),
            (height * detection.bounding_box[1]) + (height * detection.bounding_box[3])
        )

        try:
            crop = img.crop(crop_box)
        # sometimes annotations are from right to left, so coordinates order changes
        except ValueError:
            crop_box = (
                (width * detection.bounding_box[0]),
                (height * detection.bounding_box[1]) + (height * detection.bounding_box[3]),
                (width * detection.bounding_box[0]) + (width * detection.bounding_box[2]),
                (height * detection.bounding_box[1]),
            )
            crop = img.crop(crop_box)

        if self.use_class_subset:
            label = self.labels_map_rev[map_class_to_subset(detection.label)]
        else:
            label = self.labels_map_rev[detection.label]

        if self.transforms:
            if self.use_16bit:
                crop = np.asarray(crop, dtype=np.float32) / 2 ** 16
                crop = Image.fromarray(crop)

            # crop = Image.merge('RGB', (crop, crop, crop))
            crop = self.transforms(crop)

        return crop, label


class SailingLargestCropDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            fo_dataset,
            transform: torchvision.transforms = None,
            ground_truth_label: str = GROUND_TRUTH_LABEL,
            use_16bit: bool = False,
            use_class_subset: bool = False,
    ):
        self.samples = fo_dataset

        self.transforms = transform
        self.img_paths = self.samples.values("filepath")
        self.ground_truth_label = ground_truth_label
        self.use_class_subset = use_class_subset
        self.use_16bit = use_16bit

        all_classes = self.samples.distinct(f'{self.ground_truth_label}.detections.label')
        if self.use_class_subset:
            self.classes = {map_class_to_subset(c) for c in all_classes}
        else:
            self.classes = all_classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        if self.use_16bit:
            img = Image.open(img_path).convert("I")
        else:
            img = Image.open(img_path).convert("L")

        # crop detection bounding box from image
        # for now i just crop out the largest detection per image
        # TODO clean this up in the future
        largest_crop = None
        largest_crop_label = None
        largest_crop_area = 0

        for detection in sample[self.ground_truth_label].detections:
            if abs(detection.area) > largest_crop_area:
                width = sample.metadata.width
                height = sample.metadata.height
                crop_box = (
                    (width * detection.bounding_box[0]),
                    (height * detection.bounding_box[1]),
                    (width * detection.bounding_box[0]) + (width * detection.bounding_box[2]),
                    (height * detection.bounding_box[1]) + (height * detection.bounding_box[3])
                )

                try:
                    crop = img.crop(crop_box)
                # sometimes annotations are from right to left, so coordinates order changes
                except ValueError:
                    crop_box = (
                        (width * detection.bounding_box[0]),
                        (height * detection.bounding_box[1]) + (height * detection.bounding_box[3]),
                        (width * detection.bounding_box[0]) + (width * detection.bounding_box[2]),
                        (height * detection.bounding_box[1]),
                    )
                    crop = img.crop(crop_box)

                if self.use_class_subset:
                    label = self.labels_map_rev[map_class_to_subset(detection.label)]
                else:
                    label = self.labels_map_rev[detection.label]
                largest_crop = crop
                largest_crop_label = label
                largest_crop_area = detection.area

        if self.transforms:
            # TODO make this generic for input channels
            if self.use_16bit:
                largest_crop = np.asarray(largest_crop, dtype=np.float32) / 2 ** 16
                largest_crop = Image.fromarray(largest_crop)

            largest_crop = Image.merge('RGB', (largest_crop, largest_crop, largest_crop))
            largest_crop = self.transforms(largest_crop)

        return largest_crop, torch.as_tensor(largest_crop_label)
