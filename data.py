import torch
import os
import torchvision
from torchvision import transforms

DEFAULT_DATA_DIR = './data'

# from https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)


def get_dataloader(name: str, train: bool = True, **kwargs) -> torch.utils.data.DataLoader:
    """
    Returns the dataloader for a given dataset.
    :return:
    """
    if name == 'cifar10':
        return _get_cifar10(train, **kwargs)
    raise NotImplementedError(f'No such dataloader: {name}')


def _get_cifar10(train: bool, **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=_default_cifar10_transforms() if train else _default_transforms((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    trainloader = torch.utils.data.DataLoader(trainset, shuffle=train, num_workers=os.cpu_count(), **kwargs)
    return trainloader


def _default_cifar10_transforms():
    return _default_transforms(CIFAR10_MEAN, CIFAR10_STD)


def _default_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
