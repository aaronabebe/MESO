import argparse
import json
import os
import pprint
import time

import torch
import torch.nn as nn
import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

from models.models import get_eval_model
from utils import get_args, get_model_embed_dim, TENSORBOARD_LOG_DIR, fix_seeds, get_experiment_name, eval_accuracy
from data import get_dataloader


def main(args: argparse.Namespace):
    pprint.pprint(vars(args))

    fix_seeds(args.seed)
    device = torch.device(args.device)

    experiment_name = get_experiment_name(args)
    os.makedirs(f'{TENSORBOARD_LOG_DIR}/dino/{time.ctime()[:10]}', exist_ok=True)
    output_dir = f'{TENSORBOARD_LOG_DIR}/dino/{time.ctime()[:10]}/{experiment_name}'
    writer = SummaryWriter(output_dir)
    writer.add_text("args", json.dumps(vars(args)))

    model = get_eval_model(
        args.model,
        args.device,
        args.dataset,
        path_override=args.ckpt_path,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None,
        load_remote=args.wandb
    )
    embed_dim = get_model_embed_dim(model, args.model)
    if hasattr(model, "fc") and type(model.fc) != nn.Identity:
        model.fc = nn.Identity()

    if hasattr(model, "head") and type(model.head) != nn.Identity:
        if hasattr(model.head, "fc"):
            model.head.fc = nn.Identity()
        else:
            model.head = nn.Identity()

    model.to(device)

    # TODO test to enable this, requires more memory but should be faster
    # cudnn.benchmark = True

    linear_classifier = LinearClassifier(embed_dim)
    linear_classifier = linear_classifier.to(device)

    train_loader = get_dataloader(
        args.dataset, train=True,
        batch_size=args.batch_size,
        subset=-1
    )
    val_loader = get_dataloader(
        args.dataset, train=False,
        batch_size=args.batch_size,
        subset=-1
    )

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.learning_rate * args.batch_size / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    n_batches = len(train_loader.dataset) // args.batch_size
    print('n_batches', n_batches)
    best_acc = 0
    start_epoch = 0

    # TODO add resume from checkpoint

    n_steps = start_epoch * args.batch_size
    for epoch in tqdm.auto.trange(start_epoch, args.epochs):
        if args.eval:
            linear_classifier.eval()
            with torch.no_grad():
                acc1s, acc5s = [], []
                for it, (images, labels) in tqdm.autonotebook.tqdm(enumerate(val_loader), desc="val batches ", position=2,
                                                      leave=False):
                    images, labels = images.to(device), labels.to(device)
                    with torch.no_grad():
                        if "vit" in args.model:
                            intermediate_output = model.get_intermediate_layers(images, 3)
                            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                        else:
                            output = model(images)
                    output = linear_classifier(output)
                    loss = nn.CrossEntropyLoss()(output, labels)
                    if linear_classifier.num_labels >= 5:
                        acc1, acc5 = eval_accuracy(output, labels, topk=(1, 5))
                    else:
                        acc1, = eval_accuracy(output, labels, topk=(1,))
                    acc1s.append(acc1)
                    acc5s.append(acc5)
                    writer.add_scalar("val/loss", loss.item(), n_steps)
                    writer.add_scalar("val/acc1", acc1, n_steps)
                    writer.add_scalar("val/acc5", acc5, n_steps)
                    if args.wandb:
                        wandb.log({"val/loss": loss.item(), "val/acc1": acc1, "val/acc5": acc5})
                    n_steps += 1
                acc1 = torch.mean(torch.stack(acc1s))
                acc5 = torch.mean(torch.stack(acc5s))

                print(f"val acc1: {acc1:.2f} acc5: {acc5:.2f}")

                if acc1 > best_acc:
                    best_acc = acc1
                    torch.save(linear_classifier.state_dict(), f"{output_dir}/best.pth")
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    save_path = f'{output_dir}/best.pth'
                    torch.save(save_dict, save_path)
                    if args.wandb:
                        artifact = wandb.Artifact(f'{args.model}_{args.dataset}_model', type='model')
                        artifact.add_file(save_path)
                        wandb.log_artifact(artifact)

                    best_acc = acc1

            linear_classifier.train()

        for it, (images, labels) in tqdm.tqdm(enumerate(train_loader), total=n_batches, desc=" batches", position=1,
                                              leave=False):
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                if "vit" in args.model:
                    intermediate_output = model.get_intermediate_layers(images, 3)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                else:
                    output = model(images)
            output = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            writer.add_scalar("train_loss", loss.item(), n_steps)
            writer.add_scalar("epoch", epoch, n_steps)
            writer.add_scalar("lr", optimizer.param_groups[0]['lr'], n_steps)
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": optimizer.param_groups[0]['lr'],
                }, step=n_steps)
            n_steps += 1

        scheduler.step()

    print("Training of the supervised linear classifier on frozen features completed.\n"
          "Top-1 test accuracy: {acc:.1f}".format(acc=best_acc))


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=10):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


if __name__ == '__main__':
    main(get_args())
