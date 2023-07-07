import math
import os
import sys

import torch
import tqdm

import wandb
from contrastive_utils import SupConLoss, ContrastiveHeadWrapper
from dino_utils import dino_cosine_scheduler, MLPHead
from models.models import get_model
from train_utils import get_data_loaders, eval_model, get_optimizer, get_writer
from utils import get_args, fix_seeds_set_flags, get_model_embed_dim


def main(args):
    device = torch.device(args.device)

    fix_seeds_set_flags(args.seed)
    writer, output_dir = get_writer(args, sub_dir='contrastive')

    if args.wandb:
        wandb.init(project="contrastive", config=vars(args))
        wandb.watch_called = False

    train_loader, train_loader_plain, val_loader_plain, val_loader_plain_subset = get_data_loaders(args)

    example_viz_img, _ = next(iter(val_loader_plain_subset))
    example_viz_img = example_viz_img.to(device)

    model = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None
    )
    embed_dim = get_model_embed_dim(model, args.model)

    model = ContrastiveHeadWrapper(model,
                                   MLPHead(in_dim=embed_dim, out_dim=args.out_dim,
                                           norm_last_layer=args.norm_last_layer))
    model = model.to(device)

    print("=> Dataloaders, model ready.")
    print(
        "=> Number of parameters of model: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    loss_function = SupConLoss().to(device)

    params = model.parameters()
    optim = get_optimizer(args, params)

    lr_schedule = dino_cosine_scheduler(
        args.learning_rate * args.batch_size / 256,
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    print("=> Loss, optimizer and schedulers ready.")

    n_batches = len(train_loader.dataset) // args.batch_size
    best_acc = 0
    start_epoch = 0

    if args.resume:
        if args.wandb:
            artifact = wandb.use_artifact(f'mcaaroni/contrastive/{args.model}_{args.dataset}_model:latest',
                                          type='model')
            path = artifact.download()
            path = f'{path}/best.pth'
        else:
            path = f'{output_dir}/best.pth'

        if os.path.isfile(path):
            print(f"=> loading checkpoint '{path}'")
            checkpoint = torch.load(path, map_location=device)
            model.load_state_dict(checkpoint['teacher'])
            optim.load_state_dict(checkpoint['optimizer'])
            loss_function.load_state_dict(checkpoint['loss'])
            start_epoch = checkpoint['epoch']
            print(f"=> Resuming from epoch {start_epoch}")
        else:
            print(f"=> no checkpoint found at '{path}'")

    n_steps = start_epoch * args.batch_size

    for epoch in tqdm.auto.trange(start_epoch, args.epochs, desc=" epochs", position=0):
        if args.eval:
            model.eval()

            teacher_knn_acc = eval_model(args, example_viz_img, n_steps, output_dir, model.backbone, train_loader_plain,
                                         val_loader_plain, writer, wandb, epoch, prefix='teacher')

            if teacher_knn_acc > best_acc:
                save_dict = {
                    'teacher': model.state_dict(),
                    'optimizer': optim.state_dict(),
                    'epoch': epoch + 1,
                    'args': args,
                    'loss': loss_function.state_dict(),
                }
                save_path = f'{output_dir}/best.pth'
                torch.save(save_dict, save_path)
                if args.wandb:
                    artifact = wandb.Artifact(f'{args.model}_{args.dataset}_model', type='model')
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)

                best_acc = teacher_knn_acc
            model.train()

        progress_bar = tqdm.auto.tqdm(enumerate(train_loader), position=1, leave=False, total=n_batches)
        for it, (images, labels) in progress_bar:
            it = len(train_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optim.param_groups):
                param_group["lr"] = lr_schedule[it]

            # TODO clean this up and move to wrapper class
            bsz = labels.shape[0]
            images = torch.cat([images[i] for i in range(args.n_local_crops + 2)], dim=0)
            images = images.to(device)

            outputs = model(images)
            splits = torch.split(outputs, bsz, dim=0)
            outputs = torch.cat([split.unsqueeze(1) for split in splits], dim=1)

            if args.method == 'simclr':
                loss = loss_function(outputs)
            elif args.method == 'supcon':
                loss = loss_function(outputs, labels.to(device))
            else:
                raise ValueError(f'Unknown method: {args.method}')

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            optim.zero_grad()
            loss.backward()
            optim.step()

            writer.add_scalar("train_loss", loss.item(), n_steps)
            writer.add_scalar("epoch", epoch, n_steps)
            writer.add_scalar("lr", optim.param_groups[0]['lr'], n_steps)
            writer.add_scalar("weight_decay", optim.param_groups[0]['weight_decay'], n_steps)
            progress_bar.set_description(f"Loss {loss.item():.3f}")
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": optim.param_groups[0]['lr'],
                    "weight_decay": optim.param_groups[0]['weight_decay'],
                }, step=n_steps)
            n_steps += 1


if __name__ == '__main__':
    main(get_args())
