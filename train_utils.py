import json
import os
import pprint
import time

import numpy as np
import torch
import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader, default_resize_transforms, get_mean_std, DinoTransforms
from fo_utils import get_dataset
from utils import LARS, get_experiment_name, TENSORBOARD_LOG_DIR
from visualize import dino_attention, grad_cam, t_sne


def get_data_loaders(args):
    mean, std = get_mean_std(args.dataset)
    train_transforms = DinoTransforms(
        args.input_size,
        args.input_channels,
        args.n_local_crops,
        args.local_crops_scale,
        args.global_crops_scale,
        local_crop_input_factor=args.local_crop_input_factor,
        mean=mean,
        std=std
    )

    if args.dataset == 'fiftyone':
        train_data, val_data = get_dataset(dataset_dir=args.fo_dataset_dir)
        train_loader = get_dataloader(
            args.dataset, transforms=train_transforms, fo_dataset=train_data,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.train_subset
        )
        train_loader_plain = get_dataloader(
            args.dataset, fo_dataset=train_data.clone(),
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.train_subset
        )
        val_loader_plain = get_dataloader(
            args.dataset, fo_dataset=val_data,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.test_subset
        )
        val_loader_plain_subset = get_dataloader(
            args.dataset, fo_dataset=val_data.clone(),
            batch_size=1,
            transforms=default_resize_transforms(128),
            subset=1
        )
    else:
        train_loader = get_dataloader(
            args.dataset, transforms=train_transforms, train=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.train_subset
        )

        train_loader_plain = get_dataloader(
            args.dataset, train=True,
            batch_size=args.batch_size,
            subset=args.test_subset
        )
        val_loader_plain = get_dataloader(
            args.dataset, train=False,
            batch_size=args.batch_size,
            subset=args.test_subset
        )

        # sample one random batch for embedding visualization
        val_loader_plain_subset = get_dataloader(
            args.dataset,
            train=False,
            batch_size=1,
            transforms=default_resize_transforms(128),
            # using a larger input size for visualization
            subset=1
        )
    return train_loader, train_loader_plain, val_loader_plain, val_loader_plain_subset


def get_optimizer(args, params):
    if args.optimizer == 'adam':
        return torch.optim.Adam(params)
    elif args.optimizer == 'sgd':
        # lr gets set by scheduler
        return torch.optim.SGD(params, lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adamw':
        return torch.optim.AdamW(params)
    elif args.optimizer == 'lars':
        return LARS(params)  # to use with convnet and large batches

    raise ValueError(f'Unknown optimizer: {args.optimizer}')


def get_writer(args, sub_dir):
    pprint.pprint(vars(args))
    experiment_name = get_experiment_name(args)
    os.makedirs(f'{TENSORBOARD_LOG_DIR}/{sub_dir}/{time.ctime()[:10]}', exist_ok=True)
    output_dir = f'{TENSORBOARD_LOG_DIR}/{sub_dir}/{time.ctime()[:10]}/{experiment_name}'
    writer = SummaryWriter(output_dir)
    writer.add_text("args", json.dumps(vars(args)))
    return writer, output_dir


def eval_model(args, example_viz_img, n_steps, output_dir, model, train_loader_plain, val_loader_plain, writer,
               wandb, epoch, prefix):
    accuracy, precision, recall, f1 = compute_knn(model, train_loader_plain, val_loader_plain)
    writer.add_scalar(f'eval/{prefix}_knn_acc', accuracy, n_steps)
    writer.add_scalar(f'eval/{prefix}_knn_precision', precision, n_steps)
    writer.add_scalar(f'eval/{prefix}_knn_recall', recall, n_steps)
    writer.add_scalar(f'eval/{prefix}_knn_f1', f1, n_steps)
    if args.wandb:
        wandb.log({
            f'{prefix}_knn_acc': accuracy,
            f'{prefix}_knn_precision': precision,
            f'{prefix}_knn_recall': recall,
            f'{prefix}_knn_f1': f1
        }, step=n_steps)

    if args.visualize and epoch % 10 == 0:
        if 'vit_' in args.model:
            orig, attentions = dino_attention(args, [model], args.patch_size, (example_viz_img,),
                                              plot=False,
                                              path=output_dir)
        else:
            orig, attentions = grad_cam(model, args.model, (example_viz_img,), plot=False, path=output_dir)

        tsne_fig = t_sne(args, model, val_loader_plain, plot=False, path=output_dir)
        if args.wandb:
            wandb.log({
                f'{prefix}_grads': [wandb.Image(img) for img in attentions],
                f'{prefix}_tsne': wandb.Image(tsne_fig)
            }, step=n_steps)

    return accuracy


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
        for imgs, y in tqdm.auto.tqdm(data_loader, position=3, leave=False, desc=f" embedding {name}"):
            imgs = imgs.to(device)
            lists[f"X_{name}"].append(backbone(imgs).detach().cpu().numpy())
            lists[f"y_{name}"].append(y.detach().cpu().numpy())

    arrays = {k: np.concatenate(l) for k, l in lists.items()}

    estimator = KNeighborsClassifier(algorithm='ball_tree')
    estimator.fit(arrays["X_train"], arrays["y_train"])
    y_val_pred = estimator.predict(arrays["X_val"])

    acc = accuracy_score(arrays["y_val"], y_val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(arrays["y_val"], y_val_pred,
                                                               average='weighted')  # TODO maybe change to macro

    return acc, precision, recall, f1
