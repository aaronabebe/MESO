import torch
import os
import glob
import argparse

TENSORBOARD_LOG_DIR = './tb_logs'
DEFAULT_DATA_DIR = './data'

# from https://stackoverflow.com/questions/66678052/how-to-calculate-the-mean-and-the-std-of-cifar10-data
CIFAR10_MEAN = (0.49139968, 0.48215841, 0.44653091)
CIFAR10_STD = (0.24703223, 0.24348513, 0.26158784)
CIFAR10_SIZE = 32

CIFAR_10_CORRUPTIONS = (
    'brightness', 'contrast', 'defocus_blur', 'elastic_transform', 'fog', 'frost', 'gaussian_blur', 'gaussian_noise',
    'glass_blur', 'impulse_noise', 'jpeg_compression', 'motion_blur', 'pixelate', 'saturate', 'shot_noise', 'snow',
    'spatter', 'speckle_noise', 'zoom_blur'
)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run training for different ML experiments')
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use.")
    parser.add_argument("--model", type=str, default='resnet50_cifar10', help="Model to use.")
    parser.add_argument("--ckpt_path", type=str, help="Override for default model loading dir when loading a model.")

    parser.add_argument("--input_size", type=int, default=32, help="Size of the input images.")
    parser.add_argument("--n_local_crops", type=int, default=8, help="Number of local crops for DINO augmentation.")
    parser.add_argument("--local_crops_scale", type=float, nargs='+', default=(0.2, 0.4),
                        help="Scale of local crops for DINO augmentation.")
    parser.add_argument("--global_crops_scale", type=float, nargs='+', default=(0.5, 1.),
                        help="Scale of global crops for DINO augmentation.")

    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help="Optimizer to use.")
    parser.add_argument("--sam", action='store_true', default=False,
                        help='Use SAM in conjunction with standard chosen optimizer.')
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum when training with SGD as optimizer.")
    parser.add_argument("--scheduler", type=str, default=None, help="Learning rate decay for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay for optimizer")
    parser.add_argument("--warmup_steps", type=float, default=3, help="Warmup steps when using cosine LR scheduler.")

    parser.add_argument("--device", type=str, default='cuda', help="Device to use.")  # mps = mac m1 device
    parser.add_argument("--eval", action='store_true', default=False, help='Evaluate model.')
    parser.add_argument("--visualize", type=str, choices=['dino_attn', 'dino_augs', 'grad_cam'],
                        help="Visualize the loss landscape of the model.")
    return parser.parse_args()


def get_experiment_name(args):
    name = f'{args.model}_e{args.epochs}_b{args.batch_size}_o{args.optimizer}_lr{args.learning_rate:.4f}_wd{args.weight_decay:4f}'
    return name


def remove_prefix(state_dict, prefix):
    return {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}


def get_latest_model_path(name):
    model_version_path = f'{TENSORBOARD_LOG_DIR}/{name}'
    latest_version = sorted(
        os.listdir(model_version_path),
        reverse=True,
        key=lambda x: int(x.split('_')[1])
    )[0]
    ckpt_path = glob.glob(f'{model_version_path}/{latest_version}/checkpoints/*.ckpt')[0]
    print(f'Loading model from ckpt: {ckpt_path}')
    return ckpt_path


def eval_accuracy(output, target, topk=(1, 5)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1)  # top-k index: size (B, k)
        pred = pred.t()  # size (k, B)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        acc = []
        for k in topk:
            correct_k = correct[:k].float().sum()
            acc.append(correct_k * 100.0 / batch_size)

        if len(acc) == 1:
            return acc[0]
        else:
            return acc


def grad_cam_reshape_transform(tensor, height=4, width=4):
    result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


def attention_viz_forward_wrapper(attn_obj):
    """
    Forward wrapper to visualize the attention maps using timm models.
    :param attn_obj:
    :return:
    """

    def forward_hook(x):
        B, N, C = x.shape
        qkv = attn_obj.qkv(x).reshape(B, N, 3, attn_obj.num_heads, C // attn_obj.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * attn_obj.scale
        attn = attn.softmax(dim=-1)
        attn = attn_obj.attn_drop(attn)
        attn_obj.attn_map = attn
        attn_obj.cls_attn_map = attn[:, :, 0, 1:]

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = attn_obj.proj(x)
        x = attn_obj.proj_drop(x)
        return x

    return forward_hook
