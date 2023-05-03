import math
import os
import sys

import torch
import tqdm
import wandb

from dino_utils import MultiCropWrapper, MLPHead, DINOLoss, clip_gradients
from dino_utils import get_params_groups, dino_cosine_scheduler, cancel_gradients_last_layer
from models.models import get_model
from train_utils import get_data_loaders, eval_model, get_optimizer, get_writer
from utils import get_args, fix_seeds, get_model_embed_dim


def main(args):
    device = torch.device(args.device)

    fix_seeds(args.seed)
    writer, output_dir = get_writer(args, sub_dir='dino')

    if args.wandb:
        wandb.init(project="dino", config=vars(args))
        wandb.watch_called = False

    train_loader, train_loader_plain, val_loader_plain, val_loader_plain_subset = get_data_loaders(args)

    example_viz_img, _ = next(iter(val_loader_plain_subset))
    example_viz_img = example_viz_img.to(device)

    student = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None,
        pretrained=args.timm,
    )

    embed_dim = get_model_embed_dim(student, args.model)

    student = MultiCropWrapper(student,
                               MLPHead(in_dim=embed_dim, out_dim=args.out_dim, norm_last_layer=args.norm_last_layer))
    student = student.to(device)

    teacher = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None,
        pretrained=args.timm,
    )
    teacher = MultiCropWrapper(teacher,
                               MLPHead(in_dim=embed_dim, out_dim=args.out_dim, norm_last_layer=args.norm_last_layer))
    teacher = teacher.to(device)

    # teacher gets student weights and doesnt learn
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False
    print("=> Dataloaders, student and teacher ready.")
    print(
        "=> Number of parameters of student: {:.2f}M".format(sum(p.numel() for p in student.parameters()) / 1000000.0))

    dino_loss = DINOLoss(
        args.out_dim,
        args.n_local_crops + 2,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs
    ).to(device)

    params = get_params_groups(student)
    optim = get_optimizer(args, params)

    lr_schedule = dino_cosine_scheduler(
        args.learning_rate * args.batch_size / 256,
        args.min_lr,
        args.epochs, len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = dino_cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(train_loader),
    )

    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = dino_cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(train_loader))
    print("=> Loss, optimizer and schedulers ready.")

    n_batches = len(train_loader.dataset) // args.batch_size
    best_acc = 0
    start_epoch = 0

    if args.resume:
        if args.wandb:
            artifact = wandb.use_artifact(f'mcaaroni/dino/{args.model}_{args.dataset}_model:latest', type='model')
            path = artifact.download()
            path = f'{path}/best.pth'
        else:
            path = f'{output_dir}/best.pth'

        if os.path.isfile(path):
            print(f"=> loading checkpoint '{path}'")
            checkpoint = torch.load(path, map_location=device)
            student.load_state_dict(checkpoint['student'])
            teacher.load_state_dict(checkpoint['teacher'])
            optim.load_state_dict(checkpoint['optimizer'])
            dino_loss.load_state_dict(checkpoint['dino_loss'])
            start_epoch = checkpoint['epoch']
            print(f"=> Resuming from epoch {start_epoch}")
        else:
            print(f"=> no checkpoint found at '{path}'")

    n_steps = start_epoch * args.batch_size

    for epoch in tqdm.auto.trange(start_epoch, args.epochs, desc=" epochs", position=0):
        if args.eval:
            student.eval()
            teacher_knn_acc = eval_model(args, example_viz_img, n_steps, output_dir, student.backbone,
                                         train_loader_plain, val_loader_plain, writer, wandb, epoch, prefix='teacher')

            if teacher_knn_acc > best_acc:
                save_dict = {
                    'student': student.state_dict(),
                    'teacher': teacher.state_dict(),
                    'optimizer': optim.state_dict(),
                    'epoch': epoch + 1,
                    'args': args,
                    'dino_loss': dino_loss.state_dict(),
                }
                save_path = f'{output_dir}/best.pth'
                torch.save(save_dict, save_path)
                if args.wandb:
                    artifact = wandb.Artifact(f'{args.model}_{args.dataset}_model', type='model')
                    artifact.add_file(save_path)
                    wandb.log_artifact(artifact)

                best_acc = teacher_knn_acc
            student.train()

        progress_bar = tqdm.auto.tqdm(enumerate(train_loader), position=1, leave=False, total=n_batches)
        for it, (images, _) in progress_bar:
            it = len(train_loader) * epoch + it  # global training iteration
            for i, param_group in enumerate(optim.param_groups):
                param_group["lr"] = lr_schedule[it]
                if i == 0:  # only the first group is regularized
                    param_group["weight_decay"] = wd_schedule[it]

            images = [img.to(device) for img in images]

            teacher_outputs = teacher(images[:2])
            student_outputs = student(images)

            loss = dino_loss(student_outputs, teacher_outputs, epoch)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            optim.zero_grad()
            loss.backward()
            cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            clip_gradients(student, args.clip_grad)
            optim.step()

            with torch.no_grad():
                m = momentum_schedule[it]
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul(m).add_((1 - m) * param_q.detach().data)

            writer.add_scalar("train/train_loss", loss.item(), n_steps)
            writer.add_scalar("train/epoch", epoch, n_steps)
            writer.add_scalar("train/teacher_momentum", m, n_steps)
            writer.add_scalar("train/lr", optim.param_groups[0]['lr'], n_steps)
            writer.add_scalar("train/weight_decay", optim.param_groups[0]['weight_decay'], n_steps)
            progress_bar.set_description(f"Loss: {loss.item():.2f}")
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": optim.param_groups[0]['lr'],
                    "weight_decay": optim.param_groups[0]['weight_decay'],
                    "teacher_momentum": m,
                }, step=n_steps)
            n_steps += 1


if __name__ == '__main__':
    main(get_args())
