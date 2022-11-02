# SSL_MFE

Self-Supervised Learning for Robust Maritime IR/Vision Feature Extraction

## TODOs

- [ ] try loss landscapes with pre-trained model and compare to reference
- [ ] implement cifar10-c evaluation
- [ ] try out whole pipeline for simple model
- [ ] implement visualization for attention maps

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
