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

- [ ] few shot learning?
- [x] representation learning?
- [ ] zero shot?
- [ ] depthwise conv (used in mobilenet)
- [ ] FLOPs
- [ ] layernorm
- [ ] DINO teacher centering?
- [ ] JFT-300M
- [ ] Inception style pre-processing
- [ ] GPaCo/PaCo
- [ ] t-SNE
- [ ] KL-divergence

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
- [ ] cleanup code and push to repo
- [ ] implement CIFAR10-C eval in pipeline
- [ ] understand/implement code for visualizing attention maps
- [ ] train ResNet

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

