import argparse
import pprint

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.nn import functional as F
from tqdm import tqdm

from data import get_dataloader
from models import get_eval_model
from utils import eval_accuracy, CIFAR_10_CORRUPTIONS, reshape_for_plot, CIFAR10_STD, CIFAR10_MEAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default='resnet50_cifar10', help="Model to use.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use.")  # mps = mac m1 device
    parser.add_argument("-c", "--corruption", type=str, default=None,
                        help="Corruption type to evaluate.")
    return parser.parse_args()


def compute_embeddings(backbone, data_loader):
    device = next(backbone.parameters()).device

    embs_l = []
    imgs_l = []
    labels = []

    for img, y in data_loader:
        img = img.to(device)
        embs_l.append(backbone(img).detach().cpu())
        imgs_l.append(((img * CIFAR10_STD[1]) + CIFAR10_MEAN[1]).cpu())  # undo norm
        labels.extend([data_loader.dataset.classes[i] for i in y.tolist().reverse()])

    embs = torch.cat(embs_l, dim=0)
    imgs = torch.cat(imgs_l, dim=0)

    return embs, imgs, labels


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

    estimator = KNeighborsClassifier()
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)

    return acc


def main(args: argparse.Namespace):
    device = torch.device(args.device)

    model = get_eval_model(args.model)
    model.to(device)

    if args.corruption:
        corruption = [args.corruption]
    else:
        corruption = CIFAR_10_CORRUPTIONS

    result = {k: {} for k in corruption}
    for cname in corruption:
        print(f'Evaluating corruption: {cname}...')
        eval_dl = get_dataloader(name='cifar10-c', cname=cname)
        losses = []
        accs = []
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(eval_dl)):
                x = x.to(device)
                y = y.to(device, dtype=torch.int64)

                y_hat = model(x)
                loss = F.cross_entropy(y_hat, y)
                acc, _ = eval_accuracy(y_hat, y, topk=(1, 5))

                accs.append(acc.item())
                losses.append(loss.item())

        result[f'{cname}']['acc'] = np.mean(np.array(accs))
        result[f'{cname}']['loss'] = np.mean(np.array(losses))

    # result['avg_acc'] = np.mean([result[k]['acc'] for k, v in result.items() if 'acc' in result[k]])
    # result['avg_loss'] = np.mean([result[k]['loss'] for k, v in result.items() if 'loss' in result[k]])

    pprint.pprint(result)


if __name__ == '__main__':
    main(parse_args())
