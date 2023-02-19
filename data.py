import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import RandomSampler, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from fo_utils import GROUND_TRUTH_LABEL, map_class_to_subset, SAILING_CLASSES_V1
from utils import CIFAR_10_CORRUPTIONS, DEFAULT_DATA_DIR, CIFAR10_MEAN, CIFAR10_STD, CIFAR10_SIZE, MNIST_STD, \
    MNIST_MEAN, FASHION_MNIST_STD, FASHION_MNIST_MEAN, CIFAR10_LABELS, FASHION_MNIST_LABELS


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
    raise NotImplementedError(f'No such dataloader: {name}')


def get_class_labels(dataset_name):
    if dataset_name == 'cifar10':
        return CIFAR10_LABELS
    elif dataset_name == 'mnist':
        return list(range(10))
    elif dataset_name == 'fashion-mnist':
        return FASHION_MNIST_LABELS
    elif dataset_name == 'fiftyone':
        return SAILING_CLASSES_V1
    raise NotImplementedError(f'No such dataset: {dataset_name}')


def get_mean_std(dataset):
    if dataset == 'cifar10':
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'mnist':
        return MNIST_MEAN, MNIST_STD
    elif dataset == 'fashion-mnist':
        return FASHION_MNIST_MEAN, FASHION_MNIST_STD
    elif dataset == 'fiftyone':
        return FASHION_MNIST_MEAN, FASHION_MNIST_STD
        # return MNIST_MEAN, MNIST_STD
        # return SAILING_MEAN, SAILING_STD
    raise NotImplementedError(f'No such dataset: {dataset}')


def default_cifar10_transforms():
    return default_transforms(CIFAR10_SIZE, *get_mean_std('cifar10'))


def default_mnist_transforms():
    return default_transforms(CIFAR10_SIZE, *get_mean_std('mnist'))


def default_fashion_mnist_transforms():
    return default_transforms(CIFAR10_SIZE, *get_mean_std('fashion-mnist'))


def default_fifty_one_transforms():
    return default_transforms(224, *get_mean_std('fiftyone'))


def default_transforms(input_size, mean=None, std=None):
    t = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    if mean or std:
        t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


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
    trainset = FiftyOneTorchDataset(
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


def flip_and_color_jitter():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
            p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
    ])


def random_gaussian_blur(p):
    return transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=p)


def normalize(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class DinoTransforms:
    def __init__(
            self, input_size, input_channels, local_crops_number, local_crops_scale, global_crops_scale,
            local_crop_input_factor=2,
            mean=CIFAR10_MEAN,
            std=CIFAR10_STD
    ):
        self.local_crops_number = local_crops_number

        # we want to use different transforms for the grayscaled images
        if input_channels == 1:
            flip_and_jitter = transforms.RandomHorizontalFlip(p=0.5)
        else:
            flip_and_jitter = flip_and_color_jitter()

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(1.0),
            normalize(mean, std),
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(0.1),
            # deactivating for now since we're working with grayscale
            # transforms.RandomSolarize(170, p=0.2),
            normalize(mean, std),
        ])

        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(input_size // local_crop_input_factor, scale=local_crops_scale,
                                         interpolation=InterpolationMode.BICUBIC),
            flip_and_jitter,
            random_gaussian_blur(p=0.5),
            normalize(mean, std),
        ])

    def __call__(self, img):
        all_crops = [self.global_transfo1(img), self.global_transfo2(img)]
        for _ in range(self.local_crops_number):
            all_crops.append(self.local_transfo(img))
        return all_crops


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


class FiftyOneTorchDataset(torch.utils.data.Dataset):
    def __init__(self, fo_dataset, transform: torchvision.transforms = None,
                 ground_truth_label: str = GROUND_TRUTH_LABEL, use_class_subset: bool = False):
        self.samples = fo_dataset

        self.transforms = transform
        self.img_paths = self.samples.values("filepath")
        self.ground_truth_label = ground_truth_label
        self.use_class_subset = use_class_subset

        all_classes = self.samples.distinct(f'{self.ground_truth_label}.detections.label')
        if self.use_class_subset:
            self.classes = {map_class_to_subset(c) for c in all_classes}
        else:
            self.classes = all_classes
            assert len(self.classes) == len(SAILING_CLASSES_V1), "Classes are not the same"

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = Image.open(img_path).convert("L")  # TODO handle 16bit and 8bit grayscale correctly while loading

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

                # crop = np.array(crop, dtype=np.float64)
                if self.transforms:
                    crop = self.transforms(crop)

                if self.use_class_subset:
                    label = self.labels_map_rev[map_class_to_subset(detection.label)]
                else:
                    label = self.labels_map_rev[detection.label]
                largest_crop = crop
                largest_crop_label = label
                largest_crop_area = detection.area

        return largest_crop, torch.as_tensor(largest_crop_label)
