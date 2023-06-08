# Quick Research Notes

## 27.09.2022

Checking out DINO paper and references, ViT network architectures as well as MaskDINO afterwards.
Also checked out ResNet Paper again, to remember everything about skip connections.

### DINO

DINO is not the same as contrastive learning. It uses only positives from the same image with self-distillation.
Probably a good idea to compare it to contrastive learning and show the pros/cons.
It can be used with any kind of backbone, not only ViT.

- uses student/teacher to prevent collapse
- ema: exponential moving average
- centering keeps the logits of the teacher in range
- crossentropy and softmax between teacher and student instead of inner product etc. (to force them to come up with
  classes and definitions?)

## 28.09.2022

Still on the same tasks as yesterday. Also checking out DETR now.

!Important!
MaskDINO has nothing to do with DINO by FAIR. They just randomly used the same name.

MaskDINO improves DETR (Object Detection Transformers), by masking and allows semantic segmentation.

### GPaCo

Found an interesting paper about GPaCo (Generalized Parametrized Contrastive Learning), which improves
upon Self-Supervised Contrastive Learning.
-> long-tailed recognition
-> CIFAR10-LT as baseline?
-> this vs. DINO?
-> check out the source-code, it's available on
Github: [Link](https://github.com/dvlab-research/Parametric-Contrastive-Learning)

Interesting influental paper from 2006:
Hinton, G. and Salakhutdinov, R. (2006). Reducing the dimensionality of data with neural networks. Science, 313(5786):
504 – 507.

Have to check out paper:
BYOL (Bootstrap your own latent)
EsVit (Effifcient self-supervised vision transformers for representation learning)

For evaluation and comparison, may use normalized loss landscapes?
See: Li, H., Xu, Z., Taylor, G., Studer, C., and Goldstein, T. (2017). Visualizing the loss landscape of neural nets.

What are

- [x] few shot learning?
- [x] representation learning?
- [x] zero shot?
- [x] depthwise conv (used in mobilenet)
- [x] FLOPs
- [x] layernorm
- [x] DINO teacher centering?
- [ ] JFT-300M
- [ ] Inception style pre-processing
- [x] GPaCo/PaCo
- [x] t-SNE
- [x] KL-divergence

Also check out ConvNext paper and Momentum Contrast.

BYOL is older than DINO. They both use a self-supervised approach with 2 models.

### ConvNext

Step by step modernization of conv nets.

training (76.1% -> 78.8%):

- 90 -> 300 epochs
- AdamW optim (?)
- new augments: Mixup, Cutmix, RandAugment, Random Erasing,
- regularization: Stochastic Depth, Label Smoothing

macro design (78.8 -> 80.6%):

- (3, 4, 6, 3) -> (3, 3, 9, 3)
- (7, 7, 2) kernel -> (4, 4, 4) kernel)
- resnext design (?): width 64 -> 96

micro design ():

- RELU for each conv -> one GELU at the end
- BN -> LN

Check out SwinTransformer paper!

## 29.09.2022

Will look at ResNet paper, loss landscapes paper and maybe EsVit paper.
-> Only managed to read up on ResNet today.

## 01.10.2022

### Loss Landscapes

Could be used to visualize and compare models with and without pre-training. Another paper does the same thing, but only
for ViT's not for CNN's
(Contrastive Learning Rivals MIM in Fine-tuning via FD).

### SimCLRv2

uses self-supervised contrastive pre-training

## 02.10.2022

Trying to setup Tensorflow for M1 Macs on my old work PC to try out some basic
ResNet training with CIFAR10, try out tensorboard and get up to date with
Keras etc.
I opted into using a ResNet50 by TF and training it for 50epochs, which took about ~2h18mins.
I got some weird behaviour for the eval-accuracy, didn't go above 54% probably not enough regularization?
Or some bug in my implementation, because I just used the built in evaluation of `keras.Model.fit()` - does it even have
such a thing?
I probably should retry and add my own evaluation metrics.

Also I found this interesting dataset called Imagenette by Jeremy Howard [Github](https://github.com/fastai/imagenette).
It contains a subset of ImageNet with either 10 easily classifiable classes, 10 hard classifiable classes, or
10 classes with less labels or everything with noisy labels as well.
Could probably be a good starting point, to try out algorithms.

## 03.10.2022

Found interesting dataset with 20K background images (Bg20K): [Github Repo](https://github.com/JizhiziLi/GFM)
Today I want to start verbalizing a plan for the thesis, also maybe create a presentation to sum up some research ideas.

How can we measure robustness between two different approaches?
With or without pre-training?

### Using Videos to evaluate image model robustness

In this work they show that "Natural robustness" from different neighboring video frames highly correlates with
accuracy of the model compared to other adverserial transforms.
Translation, Saturation, Hue and Brightness have the highest cross-correlation with "natural robustness" from videos.

They say that no single adversarial or regularization technique improves robustness. -> Hypothesis: Maybe DINO/other
pretraining does?

### Understanding Robustness of Transformers for Image Classification

They compare different robustness benchmarks on ImageNet-C, ImageNet-A, ImageNet-R evaluation datasets.
They find that ViT only outperform comparable ResNets on robustness, when pre-trained on very large image datasets (
ImageNet-21k, JFT-300M)

### When Vision Transformers outperform ResNets without pre-training or strong data augmentations

They show that using a SAM (sharpness aware minimizer), they can improve the loss landscape and make the model more
robust.
This works especially well for ViT's, who usually need large pre-training sets to match ResNet's. Using a SAM they can
match equally sized ResNet's without any pre-training.
Using ResNet's with SAM also improves their accuracy, whereas only by a small amount (ImageNet: +0.7%).
Although improves ImageNet-C by 1.9% for ResNet-50.
-> What happens with pre-training? Can we get more robustness improvement?
-> ConvNeXt's try to be more similar to ViT, so do they improve more by using SAM?

### The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization

They introduce ImageNet-R, a 200 class subset of ImageNet with 19 differents augs and pertubations.
They also introduce DeepAug an augmentation technique using NNs. Using this + AugMix they improve a Resnet-50
on ImageNet-C accuracy by ~19%.

## 05.10.2022

Erster Arbeitstag bei sea.ai.
Laptop aufgesetzt, mit David erste Ideen von voriger Woche besprochen und verifiziert.
David hatte die Idee DINO zu verwenden aber statt Local/Global Crops zu verwenden, eher vertikale Crops zu verwenden die
zb.
Shoreline/Wasser/Himmer abbilden und diese dann mit Global Crops zu kombinieren. Und das mit Bildern aus ihrem Datensatz
die
keine Objekte enthalten.

Prinzipiell besteht ihr Datensatz aus Bildern (RGB/IR) die mit Boxen annotierte Objekte enthalten.
Es gibt auch Videos. Gespeichert wird alles in einer pyDB namens fifty-one.

Zu beachten bei Infrarot Bildern, jeder Pixel kann Werte zwischen -80 und 2000 Grad darstellen, die dann aber meistens
in einen
Bereich von 10-50 Grad normalisiert werden.

Ausserdem zusammenfassende E-Mail der ersten Recherche an Ben und David verfasst.

## 06.10.2022

### Visualizing the Loss Landscape of Neural Nets

Loss Landscape v2. Schaue mir das Paper etwas genauer an und versuche Visualisierungen eines simplen selber-trainierten
Modells zu erstellen.
Vielleicht auch schon ein Vergleich mit einem pre-trained Modell.

Great explainer blogpost:
https://towardsdatascience.com/loss-landscapes-and-the-blessing-of-dimensionality-46685e28e6a4

## 11.10.2022

zotero setup

## 12.10.2022

papers

## 17.10.2022

papers

## 20.10.2022

Erstes Seminar fuer Diplomand_innen, Terminfindung fuer Proposal-Praesentation.

## 28.10.2022

Beginn Ausarbeitung Proposal.

## 01.11.2022

Training of ViT-T/8 with CIFAR10. First implementation of loss landscape visualizations.
https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#9388

## 02.11.2022

Why std/mean for ImageNet also on other datasets?

Why std/mean for input data:
https://towardsdatascience.com/how-to-calculate-the-mean-and-standard-deviation-normalizing-datasets-in-pytorch-704bd7d05f4c

SAM-ViT papers uses 10% of ImageNet training set (~120.000) and resolution of 50x50 to visualize loss landscape.

TODOs:

- [ ] eval loss landscape for pretrained ViT
- [x] cleanup code and push to repo
- [x] implement CIFAR10-C eval in pipeline
- [x] understand/implement code for visualizing attention maps
- [ ] ~~train ResNet~~

ViT-T/8 on M1 ~1.8it/s
ViT-T/8 on Lenovo ~3.5it/s
ResNet-26 on Lenovo ~2it/s
ResNet-50 on Lenovo ~1.2it/s

loss_landscapes pip lib doesnt seem to work, will try to evaluate official implementation with selftrained model now.
Then integrate it?

## 03.11.2022

Trying official loss landscapes implementation with trained ResNet from yesterday to check if it looks different.
Official impl already takes way longer to calculate.
Uses whole CIFAR10 to calculate each point of the vis grid.

## 04.11.2022

Getting evaluation of cifar10-c to work, also evaluating loss landscape of my trained ResNet.

Somehow it looks very smooth, maybe the ResNet50 provided by TIMM works to well on CIFAR10?
-> it probably looks so smooth because of the low resolution I used! the low resolution acts as a low pass filter on the
more detailed landscape values.
But that is a big problem, reference paper uses resolution of 51x51 which would take approx. 51h :-(
-> i will try to run it on colab to see if i get better performance!

I think I will now try to get the landscape for a MLP or other bad model to see if anything can make it more bumpy.
Evaluation still takes a very long time -> loss landscape with official impl for 15x15 grid takes 2h30m on Lenovo.

Evaluation metric for CIFAR10C
[see this article on medium](https://shaktiwadekar.medium.com/evaluate-robustness-of-convolutional-neural-networks-cnns-with-cifar100-c-and-cifar10-c-datasets-15ab3592f2fa)

- Robustness Accuracy!

## 07.11.2022

looking for further visualization methods:
found cool library [grad_cam](https://github.com/jacobgil/pytorch-grad-cam)
also found interesting paper on "reliable scaling laws"

## 08.11.2022

try to implement attention map visualization similar to DINO.
implementation [example](https://github.com/rwightman/pytorch-image-models/discussions/1232)

## 09.11.2022

cleaned up code for different visualizations and model loading.
fixed code for 3D surface creation in official loss landscapes library.
looks even more wrong now, compared to the example from the official repo
maybe because I used Adam instead of SGD? -> probably a good idea to replicate the official version first,
to see if it even works?

![resnet50 loss landscape](../tb_logs/resnet50_cifar10/version_0/checkpoints/epoch=19-step=7820.ckpt_weights_xignore=biasbn_xnorm=filter_yignore=biasbn_ynorm=filter.h5_%5B-1.0,1.0,15%5Dx%5B-1.0,1.0,15%5D.h5_train_loss_3dsurface.svg)

creating loss landscape for VIT-T/8, seems to be a bit faster than ResNet50 due to less params
-> one loss sample takes ~37s

check out talk about LL [youtube link](https://www.youtube.com/watch?v=98xPveYSMv4)

## 13.11.2022

trying to train Resnet50 on google colab to check if its faster
loss landscapes still don't work correctly i think, or they are just always extremely smooth? makes no sense
i also tried running on the test set and getting the test set loss, also tried with a very small stupid net,
which i assumed would lead to bumpier loss surface, but in all cases we get no change

trained resnet50 for 300 epochs on colab, worked waaaay faster, around 13it/s
-> will prob use colab for training from now on

also visualized loss landscape, but it still looks like all the other ones -> still no idea why

will train a ViT-T overnight on 300 epochs as well, to compare and evaluate them, next up probably SAM/eSAM and ConvNeXt
training

also finished first draft of proposal today

## 14.11.2022

evaluating different model trainings including resnets, vits, found bug in
loss calculation, which led to weird eval results, hopefully fixed now,
reading some new papers as well, SimCLR, MAE
also starting with DINO training implementation

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

## 16.11.2022

starting with implementation for dino training for cifar10, maybe imagenet-tiny afterwards

## 18.11.2022

first dino implementation without warmup/proper scheduling doesnt converge, loss is stuck at around 6.35
tried out adapted official implementation with cifar10, which seems to work, stopped training after 40 epochs
-> will try to implement warmup and proper scheduling now
-> also will try to implement official attn visualization, because mine looks very different/not as good

## 19.11.2022

implements warmup and proper scheduling, loss looks better now for short training run, got memory problems now due to
batch sizes
starting on convnext implementation now

## 23.11.2022

implemented convnext, training now, also implemented official attention visualization, which looks very different

kleines datenset overfitten zum testen der implementierung

loss landscapes: mnist versuchen + mpi versuchen

vincze wegen tu hardware schreiben

## 24.11.2022

setup wandb to log
testing different training configurations and datasets for a few epochs in colab to see if everything seems to work
correctly

checking andrej karpathys training recipe for reference [link to his blog](http://karpathy.github.io/2019/04/25/recipe/)

## 03.12.2022

idea: maybe i could use a existing or pre-trained model to label the data for the self supervised task?
-> depends on the data and the model though

## 04.12.2022

working on training with smaller convnexts (atto/femto)

## 10.12.2022

retraining good performing older models on colab pro, implemented linear probing

## 11.01.2023

training ViT/ConvNext
trying to get data from Sea.ai fiftyone server, a lot of troubleshooting

setting up new pc from ben with better gpu

## 12.01.2023

retraining MobileNetv3 on new pc, using LARS optimizer getting better results than AdamW
also retraining convnext_pico now, also works great with LARS
currently running with 500 epoch trainings, convnext kNN is still increasing -> try 1000 epochs next?

- [ ] try one larger version of mobilenetv3?
- [x] try convnextv2_pico?
- [ ] try mobilevitv2_100, weight decay 0.05, min_lr 1e-6, lr 0.02, smaller batch size?

## 13.01.2023

finally got the first sample of sea.ai dataset, 20.000 datapoints
will implement new pre-processing and data loading now, then try training with DINO

i will also try to integrate contrastive loss and SimCLR into the training pipeline next

for the CIFAR10 models, im still missing linear evaluation on the best ones, and then evaluation on CIFAR10-C
prob will do ViT-Tiny, convnext_pico and mobilevitv3_small_100

## 18.01.2023

getting started with fiftyone integration, exploring dataset
thinking about using fivecrop transform for DINO

for the beginning only filter large detection crops as image and use those

number of unique classes: 32 -> maybe i can use a reduced set of classes?

```shell
{
    'ALGAE': 1,
    'BIRD': 65,
    'BOAT': 262,
    'BOAT_WITHOUT_SAILS': 456,
    'BUOY': 319,
    'CONSTRUCTION': 207,
    'CONTAINER': 51,
    'CONTAINER_SHIP': 267,
    'CRUISE_SHIP': 108,
    'DOLPHIN': 2,
    'FAR_AWAY_OBJECT': 4650,
    'FISHING_BUOY': 90,
    'FISHING_SHIP': 17,
    'FLOTSAM': 261,
    'HARBOUR_BUOY': 94,
    'HORIZON': 1,
    'HUMAN': 9,
    'HUMAN_IN_WATER': 11,
    'HUMAN_ON_BOARD': 173,
    'KAYAK': 3,
    'LEISURE_VEHICLE': 23,
    'MARITIME_VEHICLE': 936,
    'MOTORBOAT': 408,
    'OBJECT_REFLECTION': 30,
    'SAILING_BOAT': 534,
    'SAILING_BOAT_WITH_CLOSED_SAILS': 576,
    'SAILING_BOAT_WITH_OPEN_SAILS': 528,
    'SEAGULL': 3,
    'SHIP': 347,
    'SUN_REFLECTION': 11,
    'UNKNOWN': 5,
    'WATERTRACK': 105
}
```

## 19.01.2023

working on data-preprocessing

## 20.01.2023

fiftyone integration testing

## 21.01.2023

made first training run with fiftyone integration, only KNN 0.05

## 23.01.2023

checking out first training results, when only training on detection boxes of sailing boats
-> kNN accuracy of 5%, so doesnt look correct yet

should i also add segmentation evaluation? and not only linear evaluation?
also try out to integrate dino pre-trained feature weights and convnext with pre-trained feature weights

## 25.01.2023

debugging data preprocessing steps, trying to find out why performance in inital run was so bad

-> probably because of `local_crop_input_factor=2` and different `crop_scale` that was to small or to large

retrying with `local_crop_input_factor=1` and `crop_scale` in range `[0.3, 0.4]`, same as in DINO for mobilenet paper
otherwise same configuration

- [x] maybe also add labels to tSNE plot?
- [x] test different augmentations for IR images!
- [x] implement contrastive loss
- [x] implement dataloader for contrastive

- [ ] implement dataloading for 16bit images!
- [ ] check sailing dataset norm std/mean for 16bit

- [x] try out linear finetuning for pretrained convnext/dino

## 15.02.2023

implemented contrastive learning pipeline -> working good on cifar10, similar or better to DINO wrt to kNN accuracy
trying out linear eval now.

- [x] reduce amount of classes in sailing subset from 35 -> 15 maybe?
- [x] add check to guarantee same classes in train/val/test splits

## 22.02.2023

testing linear eval with DINO pretrained ViT-Base16 on sailing subset
main question is how to handle 3 channels of DINO vs 1 channel of sailing dataset

- [x] only use one channel of DINO (R/G/B)
- [x] use all 3 channels of DINO and copy sailing dataset to 3 channels
- [ ] add convolution to DINO to go from 1 to 3 channels

- [x] try out different visualizations with pretrained dino models
- [ ] cleanup model loading code for better loading of pretrained models
- [x] try out pretrained convnextv2 feature embs and compare to dino
- [ ] add visualization to linear eval code
- [x] recheck dino_attn viz code to see if it really works correctly
- [x] try out dino attn with whole image from sailing dataset
- [x] fix dino model patch size bug

#### Vit-Base16

3 channel copy: kNN 0.82
R channel: kNN 0.773334
G channel: kNN 0.8
B channel: knn 0.793334

## 02.03.2023

trying out pre-trained convnextv2 models with train_ssl DINO

results for using pre-trained models embeddings:
convnextv2-base: kNN 0.80667
convnextv2-tiny: kNN 0.78667
convnextv2-atto: kNN 0.8
convnextv2-huge: knn 0.78
mobilevitv2-200: knn 0.71334

- [x] try out grad-cam for whole images with pretrained convnextv2

## 08.03.2023

### plan for march

finish up all current tasks
try to consolidate
create timeline and summary of current progress -> find holes and missing parts to make everything sound reasonable

## 15.03.2023

re-evaluating kNN on DINO ViT-B with less clean dataset, by removing min_crop_size filter samples from 1503 -> 5363
kNN accuracy: 0.8152985074626866

also tested swinv2 base 224
kNN accuracy: 0.5093283582089553

mobilenetv3_large_100
kNN accuracy: 0.7966417910447762

mobilevitv2_200_in22ft1k
kNN accuracy: 0.6996268656716418

refactored visualization code

how to test robustness on current sailing dataset without videos? -> kNN embs, what else?

- [ ] revisit loss landscapes

tried out STEGO pretrained models to check segmentation capabilities

## 29.03.2023

- [ ] implement 16bit loading
- [ ] what about RGB data? whole dataset?

## 12.04.2023

mounting NAS in office to train on full dataset

PTH:
/mnt/fiftyoneDB/Database/Image_Data/Thermal_Images_8Bit/Trip_14_Seq_1/1286352_l.png

## 15.04.2023

setup stego training
trained with sailing20k and with massmind dataset and evaluated

## 19.04.2023

try stego training with labels
try stego training with RGB 20k

## 26.04.2023

implemented dataloader for loading all crops instead of largest crop

trying out dino/contrastive with new dataloder

- [ ] also evaluate label acc/precision/f1 for supervised contrastive?
- [x] kNN accuracy, precision, recall, f1
- [ ] implement focal loss? add to dino as loss term
- [ ] plot confusion matrix

also started setting up thesis template

## 03.05.2023

wrote first version of thesis abstract + table of contents
send consolidation )email to prof vincze

## 24.05.2023

test and prepare repo for handover

## 31.05.2023

continue writing, lay down more detailed sections for methodology

loss landscape facts:

random projections for dimensionality reduction work because of so called
Johnson–Lindenstrauss lemma

WHAT SHAPES THE LOSS LANDSCAPE OF SELF SUPERVISED LEARNING?
https://openreview.net/pdf?id=3zSn48RUO8M

## 02.06.2023

prepare everything on new pc
train small dino mobilenet with cifar
try out loss landscapes

## 03.06.2023

revisit linear eval
retrain convnext tiny dino

## 05.06.2023

convnext tiny train run died after 500 epochs, because of matplotlib memory error
restart training with convnext atto to compare
then try out vit pico configuration

-> linear finetune convnext tiny

train on sailing dataset
train on massmind dataset and evaluate on sailing?

tomorrow implement massmind data loading and cropping?
also continue writing thesis

## 06.06.2023

writing on thesis methodology, introduction, adding plots
try out knn with more neighbors, 20
try out tsne with different learning rate, 250
try out tsne with different perplexity, 20

## 07.06.2023

continuing with thesis methodology, added section for loss landscapes, added diagram for introduction of method

finished training convnext_atto cifar10 1000 epochs
also following dino evaluation settings now with neighbors=20 and distance weight

kNN accuracy:            0.5229
Precision:               0.5283266019296734
Recall:                  0.5229
F1:              0.51355132836184

- [ ] TODO fix loss landscape for convnext models

thesis

- [x] attention viz section
- [ ] feature correspondence section
- [ ] data aug section
- [x] knowledge distillation section
- [ ] parameter tuning and model selection section
- [ ] methodology introduction problems and issues
- [ ] sailing dataset section
- [ ] 
- [ ] 
- [ ] 

## 08.06.2023

continue writing thesis
data augmentation section, attention viz section, feature correspondence section


