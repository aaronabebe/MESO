import argparse

import torch
import torch.nn as nn
import tqdm
from lightning import Fabric

import wandb
from data import get_dataloader
from fo_utils import get_dataset
from models.models import get_eval_model
from train_utils import get_writer
from utils import get_args, get_model_embed_dim, fix_seeds_set_flags, eval_accuracy

N_LAST_BLOCKS_VIT_SMALL = 4
N_LAST_BLOCKS_VIT_BASE = 1


def main(args: argparse.Namespace):
    device = torch.device(args.device)
    fabric = Fabric(
        accelerator=args.device,
        precision="bf16-mixed"
    )
    fabric.launch()

    fix_seeds_set_flags(args.seed)
    writer, output_dir = get_writer(args, sub_dir='linear')

    if args.wandb:
        wandb.init(project="linear", config=vars(args))
        wandb.watch_called = False

    model = get_eval_model(
        args.model,
        args.device,
        args.dataset,
        path_override=args.ckpt_path,
        in_chans=args.input_channels,
        num_classes=0,
        img_size=args.input_size if 'vit_' in args.model else None,
        load_remote=args.wandb,
        pretrained=args.timm,
    )

    embed_dim = get_model_embed_dim(model, args.model)

    if hasattr(model, "fc") and type(model.fc) != nn.Identity:
        model.fc = nn.Identity()

    if hasattr(model, "head") and type(model.head) != nn.Identity:
        if hasattr(model.head, "fc"):
            model.head.fc = nn.Identity()
        else:
            model.head = nn.Identity()

    n_last_blocks = N_LAST_BLOCKS_VIT_BASE
    model.to(device)

    linear_classifier = LinearClassifier(embed_dim * n_last_blocks, num_labels=args.num_classes)
    linear_classifier = linear_classifier.to(device)

    if args.dataset == 'fiftyone':
        train_data, val_data = get_dataset(dataset_dir=args.fo_dataset_dir)
        train_loader = get_dataloader(
            args.dataset,
            fo_dataset=train_data,
            num_workers=args.num_workers,
            train=True,
            batch_size=args.batch_size,
            subset=args.train_subset,
        )
        val_loader = get_dataloader(
            args.dataset,
            fo_dataset=val_data,
            num_workers=args.num_workers,
            train=False,
            batch_size=args.batch_size,
            subset=args.test_subset,
        )
    else:
        train_loader = get_dataloader(
            args.dataset, train=True,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.train_subset,
        )
        val_loader = get_dataloader(
            args.dataset, train=False,
            num_workers=args.num_workers,
            batch_size=args.batch_size,
            subset=args.test_subset,
        )

    optimizer = torch.optim.SGD(
        linear_classifier.parameters(),
        args.learning_rate * args.batch_size / 256.,  # linear scaling rule
        momentum=0.9,
        weight_decay=0,  # we do not apply weight decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

    n_batches = len(train_loader.dataset) // args.batch_size
    best_acc = 0
    start_epoch = 0

    # TODO add resume from checkpoint
    fabric.setup(model, optimizer)
    train_loader, val_loader = fabric.setup_dataloaders(train_loader, val_loader)


    n_steps = start_epoch * args.batch_size
    for epoch in tqdm.auto.trange(start_epoch, args.epochs, desc=" epochs", position=0):
        if args.eval and epoch % args.eval_freq == 0:
            linear_classifier.eval()
            with torch.no_grad():
                accs, precisions, recalls, f1s = [], [], [], []
                losses = []
                progress_bar = tqdm.auto.tqdm(enumerate(val_loader), desc=" val batches", position=1, leave=False,
                                              total=len(val_loader.dataset) // args.batch_size)
                for it, (images, labels) in progress_bar:
                    with torch.no_grad():
                        if "vit_" in args.model:
                            intermediate_output = model.get_intermediate_layers(images, n_last_blocks)
                            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                        else:
                            output = model(images)

                    output = linear_classifier(output)
                    loss = nn.CrossEntropyLoss()(output, labels)
                    losses.append(loss.item())

                    acc, precision, recall, f1 = eval_accuracy(output.detach().cpu(), labels.detach().cpu())
                    accs.append(torch.as_tensor(acc))
                    precisions.append(torch.as_tensor(precision))
                    recalls.append(torch.as_tensor(recall))
                    f1s.append(torch.as_tensor(f1))

                acc = torch.mean(torch.stack(accs))
                precision = torch.mean(torch.stack(precisions))
                recall = torch.mean(torch.stack(recalls))
                f1 = torch.mean(torch.stack(f1s))
                loss = torch.mean(torch.as_tensor(losses))

                writer.add_scalar("val/acc", acc, n_steps)
                writer.add_scalar("val/precision", precision, n_steps)
                writer.add_scalar("val/recall", recall, n_steps)
                writer.add_scalar("val/f1", f1, n_steps)
                writer.add_scalar("val/loss", loss, n_steps)

                if args.wandb:
                    wandb.log({
                        "val/loss": loss,
                        "val/acc": acc,
                        "val/precision": precision,
                        "val/recall": recall,
                        "val/f1": f1,
                    })

                if acc > best_acc:
                    best_acc = acc
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

                    best_acc = acc

            linear_classifier.train()

        progress_bar = tqdm.auto.tqdm(enumerate(train_loader), position=1, leave=False, total=n_batches)
        for it, (images, labels) in progress_bar:
            with torch.no_grad():
                if "vit_" in args.model:
                    intermediate_output = model.get_intermediate_layers(images, n_last_blocks)
                    output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                else:
                    output = model(images)

            output = linear_classifier(output)
            loss = nn.CrossEntropyLoss()(output, labels)

            optimizer.zero_grad()
            fabric.backward(loss)
            optimizer.step()

            # count acc
            acc = eval_accuracy(output.detach().cpu(), labels.detach().cpu())[0]
            accs.append(torch.as_tensor(acc))

            writer.add_scalar("train/train_loss", loss.item(), n_steps)
            writer.add_scalar("train/epoch", epoch, n_steps)
            writer.add_scalar("train/lr", optimizer.param_groups[0]['lr'], n_steps)
            writer.add_scalar("train/acc", acc, n_steps)

            progress_bar.set_description(f"loss: {loss.item():.2f}")
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
