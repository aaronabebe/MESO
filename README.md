# SSL_MFE

Self-Supervised Learning for Robust Maritime IR/Vision Feature Extraction

## TODOs

- [x] try loss landscapes with pre-trained model and compare to reference
- [x] implement cifar10-c evaluation
- [x] try out whole pipeline for simple model
- [x] implement visualization for attention maps
- [x] implement dino paper first version
- [x] implement proper consine scheduling, warmup, LR decay, weight decay, temperature decay
- [ ] integrate official dino paper attn visualization
- [ ] add vit impl from dino paper

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

## Running

Run the train script with:

```shell
python train.py
```

Start tensorboard with:

```shell
tensorboard --logdir=tb_logs
```
