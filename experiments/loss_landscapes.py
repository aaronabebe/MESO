import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import loss_landscapes as ll
import os


def loss_landscape(model, model_name, data, steps=40):
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
    X = np.array([[j for j in range(steps)] for i in range(steps)])
    Y = np.array([[i for _ in range(steps)] for i in range(steps)])
    ax.plot_surface(X, Y, loss_data, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('Surface Plot of Loss Landscape')

    fig.savefig(f"./plots/{model_name}/{sub_dir_name}/{time.time()}_{len(data[0])}_surface_3D.svg")


def get_random_vectors(model):
    vector_x, vector_y = list(), list()
    weights = model.parameters()

    cast = np.array([1]).T
    for layer in weights:
        # set standard normal parameters
        # filter-wise normalization
        k = len(layer.shape) - 1
        d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)
        dist_x = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)

        vector_x.append((dist_x * (cast * np.linalg.norm(layer, axis=k))[:, np.newaxis]).reshape(d.shape))

        d = np.random.multivariate_normal([0], np.eye(1), layer.shape).reshape(layer.shape)
        dist_y = (d / (1e-10 + cast * np.linalg.norm(d, axis=k))[:, np.newaxis]).reshape(d.shape)

        vector_y.append((dist_y * (cast * np.linalg.norm(layer, axis=k))[:, np.newaxis]).reshape(d.shape))

    return weights, vector_x, vector_y


def calc_new_pos(origin, vector_x, vector_y, step_x, step_y):
    solution = [
        origin[x] + step_x * vector_x[x] + step_y * vector_y[x]
        for x in range(len(origin))
    ]
    return solution


def calc_loss_at_pos(model, data, solution):
    old_weights = model.parameters().deepcopy()

    for i, weight in enumerate(model.parameters()):
        weight.data = solution[i]

    y_hat = model(data[0])
    value = F.cross_entropy(y_hat, data[1])

    for i, weight in enumerate(model.parameters()):
        weight.data = old_weights[i]

    return value


def calc_landscape_grid(model, data, extension=1, grid_length=50):
    origin, vector_x, vector_y = get_random_vectors(model)
    space = np.linspace(-extension, extension, grid_length)

    X, Y = np.meshgrid(space, space)
    Z = []

    for i in range(grid_length):
        for j in range(grid_length):
            Z.append(calc_loss_at_pos(model, data, calc_new_pos(origin, vector_x, vector_y, X[i][j], Y[i][j])))

    Z = np.array(Z)
    return X, Y, Z


def loss_landscape_2d(model, data, vmin=0.1, vmax=10, vlevel=0.5, save=False):
    """
    Code adapted from https://github.com/artur-deluca/landscapeviz
    Plots the loss landscape in 1d.
    :return:
    """
    with torch.no_grad():
        X, Y, Z = calc_landscape_grid(model, data)

        fig, ax = plt.subplots()
        CS = ax.contour(X, Y, Z, cmap="summer", levels=np.arange(vmin, vmax, vlevel))
        ax.clabel(CS, inline=1, fontsize=8)

        if save:
            fig.savefig("./countour.svg")

        plt.show()
