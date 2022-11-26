import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.nn import functional as F

from data import get_dataloader, default_transforms, DinoTransforms, get_mean_std
from models.models import get_eval_model
from utils import grad_cam_reshape_transform, get_args, reshape_for_plot

# https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR10_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
CIFAR100_LABELS = (
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy',
    'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest',
    'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
    'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
    'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum',
    'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper',
    'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone',
    'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf',
    'woman', 'worm'
)
FASHION_MNIST_LABELS = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)


@torch.no_grad()
def grad_cam(model, model_name, data):
    """
    Visualize model reasoning via grad_cam library
    :param model:
    :param data:
    :param model_name:
    :return:
    """
    # use only one random image for now
    random_choice = random.randint(0, len(data[0]) - 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
    fig.suptitle(f'Input image class: {CIFAR10_LABELS[data[1][random_choice]]}')
    fig.tight_layout()

    if 'vit' in model_name:
        target_layer = [model.blocks[-1].norm1]
    else:
        target_layer = [model.layer4[-1]]

    input_tensor = data[0][random_choice:random_choice + 1]

    y = model(input_tensor)
    preds = [f'{CIFAR10_LABELS[i]} ({y[0][i]})' for i in y[0].argsort(descending=True)]

    cam = GradCAM(
        model=model,
        target_layers=target_layer,
        use_cuda=torch.cuda.is_available(),
        reshape_transform=grad_cam_reshape_transform if 'vit' in model_name else None,
    )
    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(data[1][random_choice])])
    input_tensor = np.transpose(input_tensor.numpy()[0], (1, 2, 0))
    ax1.imshow(input_tensor)

    visualization = show_cam_on_image(input_tensor, grayscale_cam[0, :], use_rgb=True)
    ax2.imshow(visualization)

    ax3.axis('off')
    ax3.axis('tight')
    ax3.table(
        [[p] for p in preds],
        colLabels=[f'Top {len(preds)} predictions'],
        loc='center',
    )

    sub_dir_name = 'grad_cam'
    os.makedirs(f'./plots/{model_name}/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_grad_cam.svg")

    plt.show()


@torch.no_grad()
def dino_attention(model, patch_size, data, plot=True):
    """
    Visualize the self attention of a transformer model, taken from official DINO paper.
    https://github.com/facebookresearch/dino
    """

    # use only one random image for now
    random_choice = random.randint(0, len(data[0]) - 1)

    # use only one image for now
    img = data[0][random_choice]

    patch_size = patch_size
    # make the image divisible by the patch size
    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    attentions = model.get_last_selfattention(img)

    nh = attentions.shape[1]  # number of head

    # we keep only the output patch attention
    attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
    attentions = attentions.reshape(nh, w_featmap, h_featmap)
    attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

    fig, axs = plt.subplots(1, nh + 1, figsize=(nh * 3, nh))
    if len(data) > 1:
        fig.suptitle(f"Input image class: {CIFAR10_LABELS[data[1][random_choice]]}")

    for i in range(nh):
        ax = axs[i]
        ax.imshow(attentions[i].detach().numpy())
        ax.axis("off")

    last = axs[-1]
    last.imshow(reshape_for_plot(img[0].cpu()))

    fig.tight_layout()
    sub_dir_name = 'dino_attn'
    os.makedirs(f'./plots/data/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/data/{sub_dir_name}/{time.time()}_attention.svg")

    if plot:
        plt.show()

    return img[0], attentions


@torch.no_grad()
def dino_augmentations(data):
    """
    Visualize the augmentations used in the DINO paper. Similarly to
    https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/dino/visualize_augmentations.ipynb
    """

    # use only one random image for now
    random_choice = random.randint(0, len(data[0]))
    cropped_images = [s[random_choice] for s in data[0]]

    n = int(np.ceil(len(cropped_images) ** .5))
    fig, axs = plt.subplots(n, n, figsize=(n * 3, n * 3))
    fig.suptitle(f"Input image class: {CIFAR10_LABELS[data[1][random_choice]]}")
    for i, img in enumerate(cropped_images):
        ax = axs[i // n][i % n]
        ax.imshow(reshape_for_plot(img))
        ax.axis("off")
    fig.tight_layout()
    sub_dir_name = 'dino_augs'
    os.makedirs(f'./plots/data/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/data/{sub_dir_name}/{time.time()}_attention.svg")
    plt.show()


def main(args):
    print(f'Visualizing {args.visualize} for {args.model} model...')

    if args.visualize == 'dino_attn':
        model = get_eval_model(
            args.model,
            args.device,
            path_override=args.ckpt_path,
            in_chans=args.input_channels,
            num_classes=0,
            patch_size=args.patch_size if 'vit' in args.model else None,
            img_size=32
        )
        dl = get_dataloader(args.dataset, transforms=default_transforms(args.input_size, *get_mean_std(args.dataset)),
                            train=False,
                            batch_size=args.batch_size)
        data = next(iter(dl))
        dino_attention(model, args.patch_size, data)
    elif args.visualize == 'dino_augs':
        mean, std = get_mean_std(args.dataset)
        dino_transforms = DinoTransforms(args.input_size, args.n_local_crops, args.local_crops_scale,
                                         args.global_crops_scale, mean=mean, std=std)
        dl = get_dataloader(args.dataset, transforms=dino_transforms, train=False, batch_size=args.batch_size)
        data = next(iter(dl))
        dino_augmentations(data)
    elif args.visualize == 'grad_cam':
        model = get_eval_model(
            args.model,
            args.device,
            path_override=args.ckpt_path,
            in_chans=args.input_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size if 'vit' in args.model else None,
            img_size=args.input_size
        )
        dl = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)
        data = next(iter(dl))
        grad_cam(model, args.model, data)
    else:
        raise NotImplementedError(f'Visualization {args.visualize} not implemented.')


if __name__ == '__main__':
    main(get_args())
