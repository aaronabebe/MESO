import os
import random

import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import RandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import CIFAR_10_CORRUPTIONS, DEFAULT_DATA_DIR, CIFAR10_MEAN, CIFAR10_STD, CIFAR10_SIZE, MNIST_STD, \
    MNIST_SIZE, MNIST_MEAN, FASHION_MNIST_STD, FASHION_MNIST_MEAN


def get_dataloader(name: str, transforms: torchvision.transforms = None, train: bool = True, num_workers: int = 0,
                   **kwargs) -> torch.utils.data.DataLoader:
    """
    Returns the dataloader for a given dataset.
    :return:
    """
    if name == 'cifar10':
        return _get_cifar10(train, transforms, num_workers, **kwargs)
    elif name == 'cifar10-c':
        return _get_cifar10c(transforms, num_workers, **kwargs)
    elif name == 'mnist':
        return _get_mnist(train, transforms, num_workers, **kwargs)
    elif name == 'fashion-mnist':
        return _get_fashion_mnist(train, transforms, num_workers, **kwargs)
    raise NotImplementedError(f'No such dataloader: {name}')


def get_mean_std(dataset):
    if dataset == 'cifar10':
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'mnist':
        return MNIST_MEAN, MNIST_STD
    elif dataset == 'fashion-mnist':
        return FASHION_MNIST_MEAN, FASHION_MNIST_STD
    raise NotImplementedError(f'No such dataset: {dataset}')


def default_cifar10_transforms():
    return default_transforms(CIFAR10_SIZE, *get_mean_std('cifar10'))


def default_mnist_transforms():
    return default_transforms(MNIST_SIZE, *get_mean_std('mnist'))


def default_fashion_mnist_transforms():
    return default_transforms(MNIST_SIZE, *get_mean_std('fashion-mnist'))


def default_transforms(input_size, mean=None, std=None):
    t = [transforms.Resize(input_size), transforms.ToTensor()]
    if mean or std:
        t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


def _get_fashion_mnist(train: bool, transforms: torchvision.transforms, num_workers: int,
                       **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.FashionMNIST(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_fashion_mnist_transforms(),
    )
    return torch.utils.data.DataLoader(trainset, shuffle=train, num_workers=num_workers, **kwargs)


def _get_mnist(train: bool, transforms: torchvision.transforms, num_workers: int,
               **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.MNIST(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_mnist_transforms(),
    )
    return torch.utils.data.DataLoader(trainset, shuffle=train, num_workers=num_workers, **kwargs)


def _get_cifar10c(transforms: torchvision.transforms, num_workers: int,
                  cname: str = random.choice(CIFAR_10_CORRUPTIONS), **kwargs):
    evalset = CIFAR10CDataset(
        './data/CIFAR-10-C',
        cname,
        tranform=transforms or default_cifar10_transforms()
    )
    return torch.utils.data.DataLoader(evalset, shuffle=False, num_workers=num_workers, **kwargs)


def _get_cifar10(train: bool, transforms: torchvision.transforms, num_workers: int,
                 **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=transforms or default_cifar10_transforms(),
    )
    return torch.utils.data.DataLoader(trainset, shuffle=train, num_workers=num_workers, **kwargs)


class DinoTransforms:
    def __init__(
            self, input_size, local_crops_number, local_crops_scale, global_crops_scale, mean=CIFAR10_MEAN,
            std=CIFAR10_STD
    ):
        self.local_crops_number = local_crops_number
        RandomGaussianBlur = lambda p: transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=p)

        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])

        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(1.0),
            normalize,
        ])

        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=global_crops_scale, interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(0.1),
            transforms.RandomSolarize(170, p=0.2),
            normalize,
        ])

        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=local_crops_scale,
                                         # todo: should actually be input_size // 2
                                         interpolation=InterpolationMode.BICUBIC),
            flip_and_color_jitter,
            RandomGaussianBlur(p=0.5),
            normalize,
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
