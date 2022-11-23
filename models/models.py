from functools import partial

import pytorch_lightning as pl
import timm
import torch
from timm.models import register_model, ResNet, Bottleneck
from torch import nn
from torch.nn import functional as F
from torchmetrics.functional import accuracy

from utils import remove_prefix, get_latest_model_path
from models.convnext import ConvNeXt
from models.vision_transformer import VisionTransformer


def get_model(name: str, **kwargs) -> torch.nn.Module:
    """
    Returns the model, tries to use a model implementation from timm first.
    Then uses a custom implementation once available.
    :return:
    """
    with torch.no_grad():
        return timm.create_model(name, pretrained_cfg=None, **kwargs)


def get_eval_model(name: str, path_override=None, **kwargs) -> torch.nn.Module:
    """
    Returns a self trained model from the local model directory.
    :return:
    """
    model = get_model(name, **kwargs)

    if not path_override:
        path_override = get_latest_model_path(name)

    ckpt = torch.load(path_override)
    model.load_state_dict(remove_prefix(ckpt['teacher'], 'backbone.'))
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
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
def convnext_tiny(pretrained=False, pretrained_cfg=None, **kwargs):
    return ConvNeXt(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


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


class LitNet(pl.LightningModule):
    def __init__(self, hparams):
        super(LitNet, self).__init__()
        self.args = hparams
        self.model = get_model(hparams.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y, y_hat, task='multiclass', num_classes=10, top_k=1)

        self.log('train_loss', loss)
        self.log("train_acc", acc)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y, y_hat, task='multiclass', num_classes=10, top_k=1)

        self.log(f'{stage}_loss', loss)
        self.log(f'{stage}_acc', acc)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def configure_optimizers(self):
        if self.args.sam:
            raise NotImplementedError('SAM is not working yet!')

        if self.args.optimizer == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        elif self.args.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)
        else:
            raise Exception('Please specify valid optimizer!')

        if self.args.scheduler == 'cosine':
            scheduler = timm.scheduler.CosineLRScheduler(
                optimizer=optim, t_initial=self.args.learning_rate,
                warmup_t=self.args.warmup_steps, decay_rate=self.args.lr_decay
            )
            return [optim], [{"scheduler": scheduler, "interval": "epoch"}]
        return optim

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        # timm scheduler needs epoch
        scheduler.step(self.current_epoch)
