# SSL_MFE

Self-Supervised Learning for Robust Maritime IR/Vision Feature Extraction

## TODOs

- [x] try loss landscapes with pre-trained model and compare to reference
- [x] implement cifar10-c evaluation
- [x] try out whole pipeline for simple model
- [x] implement visualization for attention maps
- [x] implement dino paper first version
- [x] implement proper consine scheduling, warmup, LR decay, weight decay, temperature decay
- [x] test dino with resnet -> convnext
- [x] integrate official dino paper attn visualization
- [x] add vit impl from dino paper
- [x] ~~fix tensorboard embedding visualizations~~ -> tsne instead
- [x] cleanup experiment tracking (naming/saving models/saving plots/folder structure/timestamps)
- [ ] improve logfiles
- [x] try out and add mobile-vit
- [ ] try out guild ai
- [ ] log representation to check for collapse of model
- [x] log loss per epoch
- [x] overfit on single batch
- [x] add continue from checkpoint
- [ ] add wandb run resuming
- [x] add train scripts for finetuning self-supervised models
- [x] add fixed seeds
- [x] implement LARS optim
- [ ] implement contrastive learning script
- [ ] try out LeVit
- [ ] implement knn from [here](https://github.com/zhirongw/lemniscate.pytorch/blob/master/test.py) and check for
  speedup

- [ ] linear probe best performing CIFAR10 model (convnext-tiny vs. vit-tiny)
- [ ] retrain convnext-tiny with larger batches for more epochs (args from `solar-rain-44` on w&b)
- [ ] compare convnext-pico/atto/femto/tiny on CIFAR10
- [ ] retrain mobilevitv2 on CIFAR10 for more epochs (args from `stilted-frost-53` on w&b)

## Notes

supervised training with contrastive loss

## Prerequisites

Download CIFAR10-C eval dataset from [here](https://zenodo.org/record/2535967#.XqZQ9hNKjIU) and put it in the `data`
folder.

## Setup

Make a venv and install the requirements:

```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Examples

Train self-supervised ViT model:

```shell
 python train_ssl.py --device cuda --model vit_tiny --eval --visualize dino_attn
```

Train self-supervised ResNet model:

```shell
python train_ssl.py --device cuda --model resnet26_dino_cifar10 --eval --visualize grad_cam
```

Train self-supervised MobileNetV3 model:

```shell
python train_ssl.py --model mobilenetv3_small_100 --batch_size 512 --eval --visualize grad_cam --optimizer lars --weight_decay 1e-5 --weight_decay_end 1e-5 --learning_rate 0.48 --norm_last_layer --wandb
```

Train self-supervised ConvNeXt on fashion-mnist:

```shell
python train_ssl.py --device cuda --model convnext_pico --optimizer adamw --epochs 200 --batch_size 128 --learning_rate 0.03 --local_crops_scale 0.2 0.5 --global_crops_scale 0.7 1. --num_classes 0 --weight_decay 1e-4 --weight_decay_end 1e-4 --dataset fashion-mnist --input_channels 1 --input_size 32 --in_dim 512
```

Visualize DINO attention maps:

```shell
python visualize.py --visualize dino_attn --model vit_tiny --ckpt_path tb_logs/dino/vit_tiny_e100_b32_oadamw_lr0.0001_wd0.040000/best.pth --input_size 480
```

### Official DINO paper args

#### ViT-S

```json
{
  "arch": "vit_small",
  "patch_size": 16,
  "out_dim": 65536,
  "norm_last_layer": false,
  "warmup_teacher_temp": 0.04,
  "teacher_temp": 0.07,
  "warmup_teacher_temp_epochs": 30,
  "use_fp16": false,
  "weight_decay": 0.04,
  "weight_decay_end": 0.4,
  "clip_grad": 0,
  "batch_size_per_gpu": 64,
  "epochs": 800,
  "freeze_last_layer": 1,
  "lr": 0.0005,
  "warmup_epochs": 10,
  "min_lr": 1e-05,
  "global_crops_scale": [
    0.25,
    1.0
  ],
  "local_crops_scale": [
    0.05,
    0.25
  ],
  "local_crops_number": 10,
  "seed": 0,
  "num_workers": 10,
  "world_size": 16,
  "ngpus": 8,
  "nodes": 2,
  "optimizer": "adamw",
  "momentum_teacher": 0.996,
  "use_bn_in_head": false,
  "drop_path_rate": 0.1
}
```

#### ResNet

```json
{
  "arch": "resnet50",
  "out_dim": 60000,
  "norm_last_layer": true,
  "warmup_teacher_temp": 0.04,
  "teacher_temp": 0.07,
  "warmup_teacher_temp_epochs": 50,
  "use_fp16": false,
  "weight_decay": 0.000001,
  "weight_decay_end": 0.000001,
  "clip_grad": 0,
  "batch_size_per_gpu": 51,
  "epochs": 800,
  "freeze_last_layer": 1,
  "lr": 0.3,
  "warmup_epochs": 10,
  "min_lr": 0.0048,
  "global_crops_scale": [
    0.14,
    1.0
  ],
  "local_crops_scale": [
    0.05,
    0.14
  ],
  "local_crops_number": 6,
  "seed": 0,
  "num_workers": 10,
  "world_size": 80,
  "optimizer": "lars",
  "momentum_teacher": 0.996,
  "use_bn_in_head": true
}
```

Train self-supervised ConvNeXt model:

```shell
python train_ssl.py --device cuda --model convnext_tiny --optimizer adamw --epochs 100 --local_crops_scale 0.2 0.5 --global_crops_scale 0.7 1. --in_dim 2048
```

Start tensorboard with:

```shell
tensorboard --logdir=tb_logs
```
