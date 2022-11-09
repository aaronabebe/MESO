import os

import numpy as np
import torch
import torchvision
from PIL import Image
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
    elif name == 'cifar10-c':
        return _get_cifar10c(**kwargs)
    raise NotImplementedError(f'No such dataloader: {name}')


def _get_cifar10c(cname: str, **kwargs):
    evalset = CIFAR10CDataset('./data/CIFAR-10-C', cname, tranform=_default_cifar10_transforms())
    return torch.utils.data.DataLoader(evalset, shuffle=False, num_workers=os.cpu_count(), **kwargs)


def _get_cifar10(train: bool, **kwargs) -> torch.utils.data.DataLoader:
    trainset = torchvision.datasets.CIFAR10(
        root=DEFAULT_DATA_DIR,
        train=train,
        download=True,
        transform=_default_cifar10_transforms() if train else _default_transforms((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    )
    return torch.utils.data.DataLoader(trainset, shuffle=train, num_workers=os.cpu_count(), **kwargs)


def _default_cifar10_transforms():
    return _default_transforms(CIFAR10_MEAN, CIFAR10_STD)


def _default_transforms(mean, std):
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])


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
