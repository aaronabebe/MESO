from typing import Any

import timm
import torch
from timm.models import register_model
from timm.models.vision_transformer import VisionTransformer


def get_model(name: str, **kwargs) -> Any:
    """
    Returns the model, tries to use a model implementation from timm first.
    Then uses a custom implementation once available.
    :return:
    """
    with torch.no_grad():
        return timm.create_model(name, **kwargs)


@register_model
def vit_tiny_cifar10(pretrained=False, **kwargs):
    """
    ViT-Tiny (Vit-T/8) for CIFAR10 training
    """
    model = VisionTransformer(img_size=32, patch_size=8, num_classes=10, embed_dim=192, depth=12, num_heads=3, **kwargs)
    if pretrained:
        print('Currently pretrained models are not available! :-(')
    return model
