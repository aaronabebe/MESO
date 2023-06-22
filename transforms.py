import torch
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from utils import CIFAR10_SIZE, CIFAR10_MEAN, CIFAR10_STD, get_mean_std


def default_cifar10_transforms():
    return default_resize_transforms(CIFAR10_SIZE, *get_mean_std('cifar10'))


def default_mnist_transforms():
    return default_resize_transforms(CIFAR10_SIZE, *get_mean_std('mnist'))


def default_fashion_mnist_transforms():
    return default_resize_transforms(CIFAR10_SIZE, *get_mean_std('fashion-mnist'))


def default_fifty_one_transforms():
    return default_resize_transforms(224, *get_mean_std('fiftyone'))


def default_resize_transforms(input_size, mean=None, std=None):
    t = [transforms.Resize((input_size, input_size)), transforms.ToTensor()]
    if mean or std:
        t.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(t)


def default_empty_transforms():
    return transforms.ToTensor()


def gauss_noise_tensor(img):
    assert isinstance(img, torch.Tensor)
    dtype = img.dtype
    if not img.is_floating_point():
        img = img.to(torch.float32)

    sigma = 8.

    print(img.shape)
    out = img + sigma * torch.randn_like(img)
    print(out.shape)

    if out.dtype != dtype:
        out = out.to(dtype)

    return out


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
    return transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0))], p=p)


def temperature_scale(img):
    scale_factor = -10
    return img + scale_factor


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
