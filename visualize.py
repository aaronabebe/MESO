import os
import time

import loss_landscapes as ll
import matplotlib.pyplot as plt
import numpy as np
import torch

# https://www.cs.toronto.edu/~kriz/cifar.html
CIFAR10_LABELS = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def loss_landscape(model, data, steps=50):
    metric = ll.metrics.Loss(torch.nn.CrossEntropyLoss(), data[0], data[1])
    loss_data = ll.random_plane(
        model=model,
        metric=metric,
        distance=10,
        steps=steps,
        normalization='filter',
        deepcopy_model=True
    )

    timestamp = time.time_ns()
    os.makedirs(f'./plots/{timestamp}', exist_ok=True)

    # plot 2D
    plt.contour(loss_data, levels=50)
    plt.savefig(f"./plots/{timestamp}/contour_2D.svg")
    plt.clf()

    # plot 3D
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    # TODO optimize this
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')
    fig.savefig(f"./plots/{timestamp}/surface_3D.svg")


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
