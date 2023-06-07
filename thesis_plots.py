from functools import partial

import plotly.graph_objects as go
import torchvision
import timm
import torch
from timm.models import register_model
import torch.nn as nn

from models.vision_transformer import VisionTransformer
from utils import get_model_embed_dim

SAILING_CLASS_DISTRIBUTION = {
    'ALGAE': 1, 'BIRD': 65, 'BOAT': 262, 'BOAT_WITHOUT_SAILS': 456, 'BUOY': 319, 'CONSTRUCTION': 207, 'CONTAINER': 51,
    'CONTAINER_SHIP': 267, 'CRUISE_SHIP': 108, 'DOLPHIN': 2, 'FAR_AWAY_OBJECT': 4650, 'FISHING_BUOY': 90,
    'FISHING_SHIP': 17, 'FLOTSAM': 261, 'HARBOUR_BUOY': 94, 'HORIZON': 1, 'HUMAN': 9, 'HUMAN_IN_WATER': 11,
    'HUMAN_ON_BOARD': 173, 'KAYAK': 3, 'LEISURE_VEHICLE': 23, 'MARITIME_VEHICLE': 936, 'MOTORBOAT': 408,
    'OBJECT_REFLECTION': 30, 'SAILING_BOAT': 534, 'SAILING_BOAT_WITH_CLOSED_SAILS': 576,
    'SAILING_BOAT_WITH_OPEN_SAILS': 528, 'SEAGULL': 3, 'SHIP': 347, 'SUN_REFLECTION': 11, 'UNKNOWN': 5,
    'WATERTRACK': 105
}


def plot_class_distribution_sailing_dataset():
    px_dict = {'class': SAILING_CLASS_DISTRIBUTION.keys(), 'count': SAILING_CLASS_DISTRIBUTION.values()}

    colors = ['lightslategray', ] * len(px_dict['class'])
    colors[10] = 'crimson'

    fig = go.Figure(data=[go.Bar(
        x=list(SAILING_CLASS_DISTRIBUTION.keys()),
        y=list(SAILING_CLASS_DISTRIBUTION.values()),
        marker_color=colors
    )])

    fig.show()


def plot_example_from_dataset(name):
    if name == 'fashion_mnist':
        trainset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=True,
            download=True,
            transform=None
        )
    else:
        trainset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            download=True,
            transform=None
        )
    trainset[345][0].show()


def print_model_params(model_name):
    with torch.no_grad():
        model = timm.create_model(model_name, pretrained_cfg=None)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000000.0
        print(f'{params:.2f}M Params')
        print(f'{get_model_embed_dim(model, model_name)}d embed dim')


@register_model
def vit_tiny(pretrained=False, patch_size=4, **kwargs):
    model = VisionTransformer(
        patch_size=patch_size, embed_dim=192, depth=9, num_heads=3, mlp_ratio=2,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


if __name__ == '__main__':
    # plot_class_distribution_sailing_dataset()
    print_model_params('mobilenetv2_050')
    # plot_example_from_dataset('cifar10')
