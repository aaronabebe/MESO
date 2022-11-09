import os
import random
import time

import loss_landscapes as ll
import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch.nn import functional as F

from utils import grad_cam_reshape_transform, attention_viz_forward_wrapper

# https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR10_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def grad_cam(model, model_name, data):
    """
    Visualize model reasoning via grad_cam library
    :param model:
    :param data:
    :param model_name:
    :return:
    """
    # use only one random image for now
    random_choice = random.randint(0, len(data[0]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(f'Input image class: {CIFAR10_LABELS[data[1][random_choice]]}')
    fig.tight_layout()

    if 'vit' in model_name:
        target_layer = [model.blocks[-1].norm1]
    else:
        target_layer = [model.layer4[-1]]

    input_tensor = data[0][random_choice:random_choice + 1]
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

    sub_dir_name = 'grad_cam'
    os.makedirs(f'./plots/{model_name}/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_grad_cam.svg")

    plt.show()


def dino_attention(model, model_name, data):
    """
    Visualize the self attention of a transformer model, similar to the DINO paper.
    https://github.com/facebookresearch/dino
    https://github.com/rwightman/pytorch-image-models/discussions/1232
    :param model:
    :param data:
    :param model_name:
    :return:
    """
    if 'vit' not in model_name:
        raise NotImplementedError('Attention visualization only works for ViT models.')

    # use only one random image for now
    random_choice = random.randint(0, len(data[0]))

    model.blocks[-1].attn.forward = attention_viz_forward_wrapper(model.blocks[-1].attn)

    # use only one image for now
    img = data[0][random_choice:random_choice + 1]
    y = model(img)
    print('\n'.join([f'{CIFAR10_LABELS[i]} ({y[0][i]})' for i in y[0].argsort(descending=True)]))

    attn_map = model.blocks[-1].attn.attn_map.mean(dim=1).squeeze(0).detach()
    cls_weight = model.blocks[-1].attn.cls_attn_map.mean(dim=1).view(4, 4).detach()

    img_resized = img[0].permute(1, 2, 0) * 0.5 + 0.5
    cls_resized = F.interpolate(cls_weight.view(1, 1, 4, 4), (32, 32), mode='bilinear').view(32, 32, 1)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 6))
    fig.suptitle(f'Input image class: {CIFAR10_LABELS[data[1][random_choice]]}')
    fig.tight_layout()

    ax1.imshow(img_resized)

    ax2.imshow(cls_resized)
    ax2.set_title('Class Attention Map')
    ax2.set_xlabel('Patch')
    ax2.set_ylabel('Patch')

    ax3.imshow(attn_map)
    ax3.set_title('Last Block Attention Map')
    ax3.set_xlabel('Head')
    ax3.set_ylabel('Patch')

    plt.show()

    sub_dir_name = 'dino'
    os.makedirs(f'./plots/{model_name}/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_attention.svg")


def loss_landscape(model, model_name, data, steps=50):
    # TODO switch impl to official repo implementation
    # TODO call via subcommand?
    metric = ll.metrics.Loss(torch.nn.CrossEntropyLoss(), data[0], data[1])
    loss_data = ll.random_plane(
        model=model,
        metric=metric,
        distance=10,
        steps=steps,
        normalization='filter',
        deepcopy_model=True
    )

    sub_dir_name = 'loss_landscape'
    os.makedirs(f'./plots/{model_name}/{sub_dir_name}', exist_ok=True)

    # plot 2D
    plt.contour(loss_data, levels=50)
    plt.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_{len(data[0])}_contour_2D.svg")
    plt.clf()

    # plot 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # TODO optimize this
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')

    fig.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_{len(data[0])}surface_3D.svg")


def show_img(img, ax, title):
    """Shows a single image."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(img[...])
    ax.set_xticks([])
    ax.set_yticks([])
    if title:
        ax.set_title(title)


def show_img_grid(imgs, labels):
    """Shows a grid of images."""
    titles = [CIFAR10_LABELS[label] for label in labels]
    n = int(np.ceil(len(imgs) ** .5))
    _, axs = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    for i, (img, title) in enumerate(zip(imgs, titles)):
        img = (img + 1) / 2  # Denormalize
        img = np.transpose(img.numpy(), (1, 2, 0))
        show_img(img, axs[i // n][i % n], title)
    plt.show()
