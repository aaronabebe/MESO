import argparse
import glob
import os
import random

import numpy as np
import torch
import torch.nn as nn
import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from fo_utils import SAILING_CLASSES_SUBSET_V1

TENSORBOARD_LOG_DIR = './tb_logs'
DEFAULT_DATA_DIR = './data'
KNN_CACHE_DIR = 'knn_cache'

# from https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)
CIFAR10_SIZE = 32

MNIST_MEAN = (0.1307,)
MNIST_STD = (0.3081,)

FASHION_MNIST_MEAN = (0.2860,)
FASHION_MNIST_STD = (0.3530,)

# SEE experiments/fo_experiments.py for more calculations
SAILING_MEAN = (0.4722,)
SAILING_STD = (0.0976,)

SAILING_16_MEAN = (0.3409,)
SAILING_16_STD = (0.0008,)

CIFAR_10_CORRUPTIONS = (
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
    'spatter', 'speckle_noise', 'zoom_blur'
)

FASHION_MNIST_LABELS = (
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
)

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


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run training for different ML experiments')
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use.")
    parser.add_argument("--train_subset", type=int, default=-1, help="Subset of dataset for training.")
    parser.add_argument("--test_subset", type=int, default=-1,
                        help="Subset of dataset for faster testing and evaluation (Default 2000)")
    parser.add_argument("--model", type=str, default='mobilenetv3_small_100', help="Model to use.")
    parser.add_argument("--ckpt_path", type=str, help="Override for default model loading dir when loading a model.")
    parser.add_argument("--compare", type=str, help="Compare visualizations of model with this model.")

    parser.add_argument("--input_size", type=int, default=64, help="Size of the input images.")
    parser.add_argument("--input_channels", type=int, default=3, help="Number of channels in the input images.")
    parser.add_argument("--num_classes", type=int, default=0, help="Number of classes in the dataset. (Defaults to 0)")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for ViT.")
    parser.add_argument('--norm_last_layer', default=True, action='store_true',
                        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")

    parser.add_argument("--out_dim", type=int, default=1024, help="Size of DINO MLPHead hidden layer output dims")
    parser.add_argument("--n_local_crops", type=int, default=8, help="Number of local crops for DINO augmentation.")
    parser.add_argument("--local_crops_scale", type=float, nargs='+', default=(0.2, 0.5),
                        help="Scale of local crops for DINO augmentation.")
    parser.add_argument("--local_crop_input_factor", type=int, default=2,
                        help="Factor by which the local crops are divided. (Default 2)")
    parser.add_argument("--global_crops_scale", type=float, nargs='+', default=(0.7, 1.),
                        help="Scale of global crops for DINO augmentation.")
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
            Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
            of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
            starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument("--momentum_teacher", type=float, default=0.9995, help="Momentum for the DINO teacher model.")
    parser.add_argument('--warmup_teacher_temp_epochs', default=10, type=int,
                        help='Number of warmup epochs for the teacher temperature (Default: 10).')
    parser.add_argument("--warmup_epochs", default=10, type=int,
                        help="Number of epochs for the linear learning-rate warm up.")

    parser.add_argument("--optimizer", type=str, default='lars', choices=['sgd', 'adam', 'adamw', 'lars'],
                        help="Optimizer to use.")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum when training with SGD as optimizer.")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer")
    parser.add_argument('--weight_decay_end', type=float, default=1e-5, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")

    parser.add_argument("--method", type=str, default='simclr', choices=['dino', 'supcon', 'simclr'])
    parser.add_argument("--device", type=str, default='cuda', help="Device to use.")  # mps = mac m1 device
    parser.add_argument("--seed", type=int, default=420, help="Fixed seed for torch/numpy/python")
    parser.add_argument("--num_workers", type=int, default=10, help="Number of dataloader workers.")
    parser.add_argument("--eval", action='store_true', default=True, help='Evaluate model during training.')
    parser.add_argument("--eval_freq", type=int, default=10, help="Evaluate model every n epochs.")
    parser.add_argument("--resume", action='store_true', default=False,
                        help='Try to resume training from last checkpoint.')
    parser.add_argument("--wandb", action='store_true', default=False, help='Log training run to Weights & Biases.')
    parser.add_argument("--timm", action='store_true', default=False, help='Use pretrained timm model.')
    parser.add_argument("--visualize", type=str, choices=['dino_attn', 'dino_proj', 'dino_augs', 'grad_cam', 'tsne'],
                        help="Visualize the model during the training.")
    parser.add_argument("--img_path", type=str, help="Path override for image for visualization.")
    parser.add_argument("--fo_dataset_dir", type=str, help="Path to fiftyone dataset.",
                        default="/home/aaron/dev/data/20000_sample_aaron")
    return parser.parse_args()


def fix_seeds_set_flags(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")


def get_experiment_name(args):
    name = f'{args.model}_{args.dataset}_e{args.epochs}_b{args.batch_size}_o{args.optimizer}'
    return name


def remove_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def get_latest_model_path(name):
    model_version_path = f'{TENSORBOARD_LOG_DIR}/{name}'
    if not os.path.exists(model_version_path):
        return None

    latest_version = sorted(
        os.listdir(model_version_path),
        reverse=True,
        key=lambda x: int(x.split('_')[1])
    )[0]
    ckpt_path = glob.glob(f'{model_version_path}/{latest_version}/checkpoints/*.ckpt')[0]
    print(f'Loading model from ckpt: {ckpt_path}')
    return ckpt_path


def eval_accuracy(output, target):
    _, preds = torch.max(output, dim=1)
    with torch.no_grad():
        acc = accuracy_score(target, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(target, preds, average='weighted', zero_division=0)
        return acc, precision, recall, f1


class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """

    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


def get_class_labels(dataset_name):
    if dataset_name == 'cifar10':
        return CIFAR10_LABELS
    elif dataset_name == 'mnist':
        return list(range(10))
    elif dataset_name == 'fashion-mnist':
        return FASHION_MNIST_LABELS
    elif dataset_name == 'fiftyone':
        return SAILING_CLASSES_SUBSET_V1
    raise NotImplementedError(f'No such dataset: {dataset_name}')


def get_mean_std(dataset):
    if dataset == 'cifar10':
        return CIFAR10_MEAN, CIFAR10_STD
    elif dataset == 'mnist':
        return MNIST_MEAN, MNIST_STD
    elif dataset == 'fashion-mnist':
        return FASHION_MNIST_MEAN, FASHION_MNIST_STD
    elif dataset == 'fiftyone':
        return SAILING_MEAN, SAILING_STD
    elif dataset == 'fiftyone-16':
        return SAILING_16_MEAN, SAILING_16_STD
    raise NotImplementedError(f'No such dataset: {dataset}')


def get_model_embed_dim(model, arch_name):
    if 'vit_' in arch_name:
        return model.embed_dim
    elif 'convnext' in arch_name:
        return model.head.in_features
    else:
        return model.num_features


def grad_cam_reshape_transform(tensor, height=8, width=8):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def reshape_for_plot(img, mean=0.5, std=0.5):
    return img.permute(1, 2, 0) * std + mean


def get_knn_cache_path(args):
    os.makedirs(KNN_CACHE_DIR, exist_ok=True)
    return f"{KNN_CACHE_DIR}/{args.model}_{args.dataset}_{args.train_subset}_{args.test_subset}{'_pretrained' if args.timm else ''}.npz"


def compute_embeddings(args, backbone, data_loader, is_training=False):
    device = next(backbone.parameters()).device

    if is_training:
        embs_l = []
        labels = []

        for img, y in tqdm.tqdm(data_loader, leave=False, desc='Computing embs'):
            img = img.to(device)
            embs_l.append(backbone(img).detach().cpu())
            labels.extend(y.tolist())

        embs = torch.cat(embs_l, dim=0)
        labels = np.array(labels)
    else:
        print('=> Loading cached embeddings...')
        cached_path = get_knn_cache_path(args)
        if os.path.exists(cached_path):
            cached = np.load(cached_path)
            embs = torch.from_numpy(cached['embs'])
            labels = cached['labels']
        else:
            raise ValueError(f'No cached embeddings for {args.dataset} and {args.model}. Please run precompute_knn.py')

    return embs, labels


def remove_head(backbone):
    if hasattr(backbone, "fc") and type(backbone.fc) != nn.Identity:
        backbone.fc = nn.Identity()
    if hasattr(backbone, "head") and type(backbone.head) != nn.Identity:
        if hasattr(backbone.head, "fc"):
            backbone.head.fc = nn.Identity()
        else:
            backbone.head = nn.Identity()
