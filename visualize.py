import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.manifold import TSNE
from torch.nn import functional as F

from data import get_dataloader, default_transforms, DinoTransforms, get_mean_std
from models.models import get_eval_model
from utils import grad_cam_reshape_transform, get_args, reshape_for_plot, CIFAR10_LABELS, compute_embeddings


def grad_cam(model, model_name, data, plot=True, path=None):
    """
    Visualize model reasoning via grad_cam library
    """
    # use only one random image for now
    device = next(model.parameters()).device

    random_choice = random.randint(0, len(data[0]) - 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    classifier_target = None
    if len(data) > 1:
        fig.suptitle(f'Input image class: {CIFAR10_LABELS[data[1][random_choice]]}')
        classifier_target = [ClassifierOutputTarget(data[1][random_choice])]

    fig.tight_layout()

    if 'vit_' in model_name:
        target_layer = [model.blocks[-1].norm1]
    elif 'convnext' in model_name or 'mobilevit' in model_name:
        target_layer = [model.stages[-1][-1]]
    else:
        target_layer = [model.layer4[-1]]

    input_tensor = data[0][random_choice:random_choice + 1]

    cam = GradCAM(
        model=model,
        target_layers=target_layer,
        use_cuda=device == 'cuda',
        reshape_transform=grad_cam_reshape_transform if 'vit_' in model_name else None,
    )
    grayscale_cam = cam(input_tensor=input_tensor, targets=classifier_target)
    input_tensor = np.transpose(input_tensor.cpu().numpy()[0], (1, 2, 0))
    ax1.imshow(input_tensor)

    visualization = show_cam_on_image(input_tensor, grayscale_cam[0, :], use_rgb=True)
    ax2.imshow(visualization)

    if not path:
        path = f"./plots/grad_cam"

    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{time.ctime()}_grads.svg")

    if plot:
        plt.show()

    plt.close()

    return input_tensor, [visualization]


@torch.no_grad()
def t_sne(model, data_loader, plot=True, path=None):
    """
    Visualize model reasoning via t-SNE
    """
    embs, _, labels = compute_embeddings(model, data_loader)
    fig = plt.figure()
    tsne = TSNE(n_components=2, random_state=123, verbose=1 if plot else 0, init='pca', learning_rate='auto')
    z = tsne.fit_transform(embs)
    plt.scatter(z[:, 0], z[:, 1], c=labels)

    if not path:
        path = f"./plots/tsne"

    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{time.ctime()}_tsne.svg")

    if plot:
        plt.show()

    return fig


@torch.no_grad()
def dino_attention(models: list[torch.nn.Module], patch_size, data, plot=True, path=None):
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

    all_attentions = []
    for i, model in enumerate(models):
        all_attentions.append(model.get_last_selfattention(img))
        attentions = all_attentions[i]

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

        if not path:
            path = f"./plots/dino_attn"

        os.makedirs(path, exist_ok=True)
        fig.savefig(f"{path}/{time.ctime()}_attention.svg")

        if plot:
            plt.show()

    plt.close()

    return img[0], all_attentions[0]


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
        models = []
        model = get_eval_model(
            args.model,
            args.device,
            args.dataset,
            path_override=args.ckpt_path,
            in_chans=args.input_channels,
            num_classes=0,
            patch_size=args.patch_size if 'vit' in args.model else None,
            img_size=32,
            load_remote=args.wandb
        )
        models.append(model)
        if args.compare:
            model2 = get_eval_model(
                args.model,
                args.device,
                args.dataset,
                path_override=args.compare,
                in_chans=args.input_channels,
                num_classes=0,
                patch_size=args.patch_size if 'vit' in args.model else None,
                img_size=32,
                load_remote=False
            )
            models.append(model2)

        dl = get_dataloader(
            args.dataset,
            transforms=default_transforms(args.input_size),
            train=False,
            subset=1,
            batch_size=args.batch_size
        )
        data = next(iter(dl))
        dino_attention(models, args.patch_size, data)

    elif args.visualize == 'dino_augs':
        mean, std = get_mean_std(args.dataset)
        dino_transforms = DinoTransforms(args.input_size, args.n_local_crops, args.local_crops_scale,
                                         args.global_crops_scale, mean=mean, std=std)
        dl = get_dataloader(
            args.dataset,
            transforms=dino_transforms,
            subset=1,
            train=False,
            batch_size=args.batch_size
        )
        data = next(iter(dl))
        dino_augmentations(data)

    elif args.visualize == 'grad_cam':
        model = get_eval_model(
            args.model,
            args.device,
            args.dataset,
            path_override=args.ckpt_path,
            in_chans=args.input_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size if 'vit_' in args.model else None,
            img_size=args.input_size if 'vit_' in args.model else None,
            load_remote=args.wandb
        )
        dl = get_dataloader(
            args.dataset,
            transforms=default_transforms(args.input_size),
            subset=1,
            train=False, batch_size=args.batch_size
        )
        data = next(iter(dl))
        grad_cam(model, args.model, data)
    elif args.visualize == 'tsne':
        model = get_eval_model(
            args.model,
            args.device,
            args.dataset,
            path_override=args.ckpt_path,
            in_chans=args.input_channels,
            num_classes=args.num_classes,
            patch_size=args.patch_size if 'vit_' in args.model else None,
            img_size=args.input_size if 'vit_' in args.model else None,
            load_remote=args.wandb
        )
        dl = get_dataloader(
            args.dataset,
            subset=-1,
            train=False, batch_size=args.batch_size
        )
        t_sne(model, dl)
    else:
        raise NotImplementedError(f'Visualization {args.visualize} not implemented.')


if __name__ == '__main__':
    main(get_args())
