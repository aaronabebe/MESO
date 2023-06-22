from functools import partial

import timm
import torch
import wandb
from timm.models import register_model
from torch import nn

from models.convnext import ConvNeXt
from models.vision_transformer import VisionTransformer
from utils import remove_prefix, get_latest_model_path, remove_head


def get_model(name: str, **kwargs) -> torch.nn.Module:
    """
    Returns the model, tries to use a model implementation from timm first.
    Then uses a custom implementation once available.
    :return:
    """
    with torch.no_grad():
        return timm.create_model(name, pretrained_cfg=None, **kwargs)


def get_eval_model(name: str, device: torch.device, dataset: str, path_override=None, load_remote: bool = False,
                   pretrained: bool = False, _remove_head: bool = False, **kwargs) -> torch.nn.Module:
    """
    Returns a self trained model from the local model directory.
    :return:
    """
    model = get_model(name, pretrained=pretrained, **kwargs)

    if pretrained:
        if name == 'vit_base':
            url = 'https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth'
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
            model.load_state_dict(state_dict)
            print('Using pretrained model from FAIR')
        elif name == 'vit_small':
            url = 'https://dl.fbaipublicfiles.com/dino/dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth'
            state_dict = torch.hub.load_state_dict_from_url(url, map_location=device)
            model.load_state_dict(state_dict)
            print('Using pretrained model from FAIR')
        else:
            print('Using pretrained model from timm')

        model.eval()
        return model

    if path_override is None:
        if load_remote:
            wandb.init()
            artifact = wandb.use_artifact(f'mcaaroni/dino/{name}_{dataset}_model:latest', type='model')
            path_override = artifact.download()
            path_override = f'{path_override}/best.pth'
        else:
            path_override = get_latest_model_path(name)

        if path_override is None:
            print('No pretrained model found. Starting with uninitialized weights.')
            return model

    print(f'Loading model from {path_override}')
    ckpt = torch.load(path_override, map_location=device)

    model.load_state_dict(remove_prefix(ckpt['student'], 'backbone.'), strict=False)
    model.eval()
    if _remove_head:
        remove_head(model)
    return model


@register_model
def vit_tiny_cifar10(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-T/8) for CIFAR10 training
    if pretrained = True then returns a Vit-T/16 pretrained on ImageNet with size 224.
    """
    return VisionTransformer(img_size=32, patch_size=8, num_classes=10, embed_dim=192, depth=12, num_heads=3, **kwargs)


@register_model
def vit_tiny(pretrained=False, patch_size=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=9, num_heads=3, mlp_ratio=2,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_pico(pretrained=False, patch_size=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=96, depth=6, num_heads=2, mlp_ratio=2,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

@register_model
def vit_small(patch_size=16, pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def vit_base(patch_size=8, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


@register_model
def convnext_tiny(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@register_model
def convnext_pico(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return ConvNeXt(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)


@register_model
def convnext_femto(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return ConvNeXt(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)


@register_model
def convnext_atto(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs):
    return ConvNeXt(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)


@register_model
def vit_tiny_dino_cifar10(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-T/8) for DINO CIFAR10 training
    if pretrained = True then returns a Vit-T/16 pretrained on ImageNet with size 224.
    """
    return VisionTransformer(img_size=32, patch_size=4, num_classes=0, embed_dim=192, depth=9, num_heads=3, mlp_ratio=2,
                             **kwargs)


@register_model
def dino_b_cifar100(pretrained=False, **kwargs):
    return VisionTransformer(img_size=224, patch_size=16, embed_dim=768, num_heads=12, depth=12,
                             **kwargs)
