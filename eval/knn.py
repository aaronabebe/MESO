import torch
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

from utils import CIFAR10_STD, CIFAR10_MEAN
from visualize import CIFAR10_LABELS


@torch.no_grad()
def compute_knn(backbone, train_loader_plain, val_loader_plain):
    device = next(backbone.parameters()).device

    data_loaders = {
        "train": train_loader_plain,
        "val": val_loader_plain,
    }
    lists = {
        "X_train": [],
        "y_train": [],
        "X_val": [],
        "y_val": [],
    }

    for name, data_loader in data_loaders.items():
        for imgs, y in data_loader:
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier(algorithm='ball_tree')
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc


@torch.no_grad()
def compute_embeddings(backbone, data_loader):
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(((img * CIFAR10_STD[1]) + CIFAR10_MEAN[1]).cpu())  # undo norm
        labels.extend([CIFAR10_LABELS[i] for i in y.tolist()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels


