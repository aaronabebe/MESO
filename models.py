import timm
import torch
from timm.models import register_model, ResNet, Bottleneck
from timm.models.vision_transformer import VisionTransformer
from torch.nn import functional as F

from utils import remove_prefix, get_latest_model_path


def get_model(name: str, **kwargs) -> torch.nn.Module:
    """
    Returns the model, tries to use a model implementation from timm first.
    Then uses a custom implementation once available.
    :return:
    """
    with torch.no_grad():
        return timm.create_model(name, pretrained_cfg=None, **kwargs)


def get_eval_model(name: str, ckpt_path=None, **kwargs) -> torch.nn.Module:
    """
    Returns a self trained model from the local model directory.
    :return:
    """
    model = get_model(name, **kwargs)

    if not ckpt_path:
        ckpt_path = get_latest_model_path(name)

    ckpt = torch.load(ckpt_path)

    # remove prefix due to PL state dict naming
    if 'dino' in name:
        model.load_state_dict(remove_prefix(ckpt, 'module.'))
    else:
        model.load_state_dict(remove_prefix(ckpt['state_dict'], 'model.'))
    model.eval()
    return model


@register_model
def vit_tiny_cifar10(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-T/8) for CIFAR10 training
    if pretrained = True then returns a Vit-T/16 pretrained on ImageNet with size 224.
    """
    return VisionTransformer(img_size=32, patch_size=8, num_classes=10, embed_dim=192, depth=12, num_heads=3, **kwargs)


@register_model
def resnet26_cifar10(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError('No pretrained ResNets :-(')

    return ResNet(block=Bottleneck, layers=[2, 2, 2, 2], num_classes=10, **kwargs)


@register_model
def dino_b_cifar100(pretrained=False, **kwargs):
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=768, num_heads=12, depth=12, num_classes=100,
                             **kwargs)


@register_model
def resnet50_cifar10(pretrained=False, **kwargs):
    if pretrained:
        raise NotImplementedError('No pretrained ResNets :-(')

    return ResNet(block=Bottleneck, layers=[3, 4, 6, 3], num_classes=10, **kwargs)


@register_model
def stupidnet_cifar10(pretrained=False, **kwargs):
    return StupidNet()


class StupidNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 10, 3)
        self.linear = torch.nn.Linear(9000, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
