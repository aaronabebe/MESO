import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from sklearn.manifold import TSNE
from torch.nn import functional as F

from data import get_dataloader, default_resize_transforms, DinoTransforms, get_mean_std, get_class_labels
from fo_utils import get_dataset
from models.models import get_eval_model
from utils import grad_cam_reshape_transform, get_args, reshape_for_plot, CIFAR10_LABELS, compute_embeddings, fix_seeds


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
    elif 'convnext_' in model_name or 'mobilevit' in model_name:
        target_layer = [model.stages[-1][-1]]
    elif 'convnextv2' in model_name:
        target_layer = [model.stages[-1].blocks[-1]]
    elif 'mobilenet' in model_name:
        target_layer = [model.blocks[-1][0].bn1]
    else:
        target_layer = [model.layer4[-1]]

    input_tensor = data[0][random_choice:random_choice + 1]

    if not input_tensor.requires_grad:
        input_tensor.requires_grad = True

    cam = GradCAM(
        model=model,
        target_layers=target_layer,
        use_cuda=device == 'cuda',
        reshape_transform=grad_cam_reshape_transform if 'vit_' in model_name else None,
    )
    grayscale_cam = cam(input_tensor=input_tensor, targets=classifier_target)
    input_tensor = np.transpose(input_tensor.cpu().detach().numpy()[0], (1, 2, 0))
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
    del cam

    return input_tensor, [visualization]


@torch.no_grad()
def t_sne(args, model, data_loader, plot=True, path=None, class_mean=False):
    """
    Visualize model reasoning via t-SNE
    """
    embs, _, labels = compute_embeddings(model, data_loader)

    fig, ax = plt.subplots()
    tsne = TSNE(
        n_components=2,
        random_state=123,
        verbose=1 if plot else 0,
        init='pca',
        perplexity=30,
        n_iter=1000,
        learning_rate='auto'
    )
    z = tsne.fit_transform(embs)

    class_names = get_class_labels(args.dataset)

    for i, label in enumerate(class_names):
        z_i = z[labels == i]
        if class_mean:
            z_i = np.mean(z_i, axis=0)
            ax.scatter(z_i[0], z_i[1], label=label, s=100)
        else:
            ax.scatter(z_i[:, 0], z_i[:, 1], label=label, alpha=0.6)

    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    ax.grid(True)

    if not path:
        path = f"./plots/tsne"

    fig.tight_layout()
    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{time.ctime()}_tsne.svg")

    if plot:
        plt.show()

    plt.close()
    return fig


@torch.no_grad()
def dino_attention(args, models, patch_size, data, plot=True, path=None, sample_size=2, avg_heads=True):
    """
    Visualize the self attention of a transformer model, taken from official DINO paper.
    https://github.com/facebookresearch/dino
    """

    # use only one random image for now
    random_choice = random.randint(0, len(data[0]) - 1)

    imgs = data[0][random_choice:random_choice + sample_size]

    for img in imgs:
        patch_size = patch_size
        # make the image divisible by the patch size
        w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        all_attentions = []
        for i, model in enumerate(models):
            attentions = model.get_last_selfattention(img)

            nh = attentions.shape[1]  # number of head

            # we keep only the output patch attention
            attentions = attentions[0, :, 0, 1:].reshape(nh, -1)
            attentions = attentions.reshape(nh, w_featmap, h_featmap)
            attentions = F.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

            all_attentions.append(attentions)

            if avg_heads:
                fig, axs = plt.subplots(1, 2, figsize=(14, 10))
                avg_attn = sum(attentions[i] * 1 / attentions.shape[0] for i in range(attentions.shape[0]))
                axs[0].imshow(avg_attn.detach().numpy())
                axs[0].axis("off")
            else:
                fig, axs = plt.subplots(1, nh + 1, figsize=(nh * 3, nh))
                for j in range(nh):
                    ax = axs[j]
                    ax.imshow(attentions[j].detach().numpy())
                    ax.axis("off")

            last = axs[-1]
            last.imshow(reshape_for_plot(img[0].cpu()))
            last.axis("off")

            if len(data) > 1:
                labels_map = get_class_labels(dataset_name=args.dataset)
                fig.suptitle(f"Input image class: {labels_map[data[1][random_choice]]}")

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
def dino_simple_projection(args, model, patch_size, data, plot=True, path=None, sample_size=2):
    # use only one random image for now
    random_choice = random.randint(0, len(data[0]) - 1)
    img = data[0][random_choice]
    print(img.shape)

    w, h = img.shape[1] - img.shape[1] % patch_size, img.shape[2] - img.shape[2] % patch_size
    img = img[:, :w, :h].unsqueeze(0)

    w_featmap = img.shape[-2] // patch_size
    h_featmap = img.shape[-1] // patch_size

    query_points = torch.tensor(
        [
            # [-.1, 0.0],
            # [.5, .8],
            [-9., -.9],
            [9., -.9],
            [.0, .0],
            [.9, .9],
            [-.9, .9],
        ]
    ).reshape(1, 5, 1, 2)

    feat = model.get_intermediate_layers(img)[0]
    feats1 = feat[:, 1:, :].reshape(feat.shape[0], h_featmap, w_featmap, -1).permute(0, 3, 1, 2)

    sfeats1 = F.grid_sample(feats1, query_points.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)

    attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)

    heatmap_intra = F.interpolate(
        attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()

    colors = np.array([
        [128, 0, 0],
        [0, 128, 0],
        [0, 0, 128],
        [128, 128, 0],
        [128, 0, 128],
    ])

    result_images = [torch.zeros((3, img.shape[2], img.shape[3])) for _ in range(5)]
    for i in range(heatmap_intra.shape[0]):
        result_images[i] = heatmap_intra[i] * torch.tensor(colors[i]).reshape(3, 1, 1)

    nr = len(result_images)
    fig, axs = plt.subplots(1, nr + 1, figsize=(nr * 5, nr))
    for j in range(nr):
        ax = axs[j]
        ax.imshow(result_images[j].detach().permute(1, 2, 0))
        ax.set_title(query_points[0, j].numpy()[0])

    last = axs[-1]
    last.imshow(reshape_for_plot(img[0].cpu()))
    last.axis("off")

    if len(data) > 1:
        labels_map = get_class_labels(dataset_name=args.dataset)
        fig.suptitle(f"Input image class: {labels_map[data[1][random_choice]]}")

    fig.tight_layout()

    if not path:
        path = f"./plots/dino_simple_projection"

    os.makedirs(path, exist_ok=True)
    fig.savefig(f"{path}/{time.ctime()}.svg")

    if plot:
        plt.show()

    plt.close()
    return img


@torch.no_grad()
def dino_augmentations(args, data):
    """
    Visualize the augmentations used in the DINO paper. Similarly to
    https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/dino/visualize_augmentations.ipynb
    """

    mean, std = get_mean_std(args.dataset)
    data = data[0]
    # use only one random image for now
    random_choice = random.randint(0, len(data[0]) - 1)
    cropped_images = [s[random_choice] for s in data]

    n = int(np.ceil(len(cropped_images) ** .5))
    fig, axs = plt.subplots(n, n, figsize=(n * 3, n * 3))
    for i, img in enumerate(cropped_images):
        ax = axs[i // n][i % n]
        ax.imshow(reshape_for_plot(img, mean[0], std[0]))
        ax.axis("off")
    fig.tight_layout()
    sub_dir_name = 'dino_augs'
    os.makedirs(f'./plots/data/{sub_dir_name}', exist_ok=True)
    fig.savefig(f"./plots/data/{sub_dir_name}/{time.time()}_augs.svg")
    plt.show()


def load_example_viz_image(image_path, input_channels, dataset, transforms):
    img = Image.open(image_path)
    if input_channels == 1:
        img = img.convert('L')
    else:
        if dataset == 'fiftyone':
            img = img.convert('L')
            img = Image.merge('RGB', (img, img, img))
        else:
            img = img.convert('RGB')

    img = transforms(img)

    # add two 0 dims to match dataloader batch
    data = torch.as_tensor(img)
    data = data.unsqueeze(0)
    data = data.unsqueeze(0)
    return data


def is_timm_compatible(model_name):
    return model_name in ['vit_', 'convnextv2']


def main(args):
    print(f'Visualizing {args.visualize} for {args.model} model...')

    fix_seeds(args.seed)

    # transforms
    if args.visualize in ['dino_attn', 'dino_proj', 'grad_cam']:
        transforms = default_resize_transforms(768)
    elif args.visualize == 'dino_augs':
        mean, std = get_mean_std(args.dataset)
        transforms = DinoTransforms(
            args.input_size, args.input_channels,
            args.n_local_crops, args.local_crops_scale,
            args.global_crops_scale, mean=mean, std=std
        )
    else:
        # use default transforms per dataset
        transforms = None

    # data
    if args.img_path is None:
        fo_dataset = None
        if args.dataset == 'fiftyone':
            fo_dataset, _ = get_dataset(dataset_dir=args.fo_dataset_dir)
        dl = get_dataloader(
            args.dataset,
            transforms=transforms,
            fo_dataset=fo_dataset,
            train=False,
            subset=args.test_subset,
            batch_size=args.batch_size
        )
        data = next(iter(dl))
    else:
        data = load_example_viz_image(args.img_path, args.input_channels, args.dataset, transforms)

    # model
    model = get_eval_model(
        args.model,
        args.device,
        args.dataset,
        path_override=args.ckpt_path,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if is_timm_compatible(args.model) else None,
        img_size=args.input_size if is_timm_compatible(args.model) else None,
        load_remote=args.wandb,
        pretrained=args.timm
    )

    # viz
    if args.visualize == 'dino_attn':
        models = []
        models.append(model)
        if args.compare:
            model2 = get_eval_model(
                args.model,
                args.device,
                args.dataset,
                path_override=args.compare,
                in_chans=args.input_channels,
                num_classes=args.num_classes,
                patch_size=args.patch_size if 'vit' in args.model else None,
                img_size=args.input_size,
                load_remote=False
            )
            models.append(model2)
        dino_attention(args, models, args.patch_size, data)

    elif args.visualize == 'dino_augs':
        dino_augmentations(args, data)

    elif args.visualize == 'dino_proj':
        dino_simple_projection(args, model, args.patch_size, data)

    elif args.visualize == 'grad_cam':
        grad_cam(model, args.model, data)

    elif args.visualize == 'tsne':
        if args.img_path is not None:
            print('tsne only works with full dataset')

        t_sne(args, model, dl)
    else:
        raise NotImplementedError(f'Visualization {args.visualize} not implemented.')


if __name__ == '__main__':
    main(get_args())
