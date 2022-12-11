import json
import math
import os
import pprint
import sys
import time

import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

import wandb
from data import get_dataloader, DinoTransforms, get_mean_std, default_transforms
from dino_utils import MultiCropWrapper, MLPHead, DINOLoss, clip_gradients
from dino_utils import get_params_groups, dino_cosine_scheduler, cancel_gradients_last_layer
from eval.knn import compute_knn
from models.models import get_model
from utils import get_args, TENSORBOARD_LOG_DIR, get_experiment_name, fix_seeds, get_model_embed_dim
from visualize import dino_attention, grad_cam, t_sne


def main(args):
    pprint.pprint(vars(args))

    fix_seeds(args.seed)
    device = torch.device(args.device)

    experiment_name = get_experiment_name(args)
    os.makedirs(f'{TENSORBOARD_LOG_DIR}/dino/{time.ctime()[:10]}', exist_ok=True)
    output_dir = f'{TENSORBOARD_LOG_DIR}/dino/{time.ctime()[:10]}/{experiment_name}'
    writer = SummaryWriter(output_dir)
    writer.add_text("args", json.dumps(vars(args)))

    if args.wandb:
        wandb.init(project="dino", config=vars(args))
        wandb.watch_called = False

    mean, std = get_mean_std(args.dataset)
    transforms = DinoTransforms(
        args.input_size,
        args.n_local_crops,
        args.local_crops_scale,
        args.global_crops_scale,
        mean=mean,
        std=std
    )

    train_loader = get_dataloader(
        args.dataset, transforms=transforms, train=True,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        subset=args.train_subset
    )
    train_loader_plain = get_dataloader(
        args.dataset, train=True,
        batch_size=args.batch_size,
        subset=args.test_subset
    )
    val_loader_plain = get_dataloader(
        args.dataset, train=False,
        batch_size=args.batch_size,
        subset=args.test_subset
    )

    # sample one random batch for embedding visualization
    val_loader_plain_subset = get_dataloader(
        args.dataset,
        train=False,
        batch_size=1,
        transforms=default_transforms(32),
        # using a larger input size for visualization
        subset=1
    )

    example_viz_img, _ = next(iter(val_loader_plain_subset))
    example_viz_img = example_viz_img.to(device)

    student = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None
    )

    embed_dim = get_model_embed_dim(student, args.model)

    student = MultiCropWrapper(student, MLPHead(in_dim=embed_dim, out_dim=args.out_dim))
    student = student.to(device)

    teacher = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit_' in args.model else None,
        img_size=args.input_size if 'vit_' in args.model else None
    )
    teacher = MultiCropWrapper(teacher, MLPHead(in_dim=embed_dim, out_dim=args.out_dim))
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
    if args.optimizer == 'adam':
        optim = torch.optim.Adam(params)
    elif args.optimizer == 'sgd':
        # lr gets set by scheduler
        optim = torch.optim.SGD(params, lr=0, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optim = torch.optim.AdamW(params)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    lr_schedule = dino_cosine_scheduler(
        args.learning_rate * args.batch_size / 128,
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

    # cifar10 trainset contains 50000 images
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
            teacher.eval()

            eval_model(args, example_viz_img, n_steps, output_dir, student, train_loader_plain,
                       val_loader_plain, writer, wandb, prefix='student')
            teacher_knn_acc = eval_model(args, example_viz_img, n_steps, output_dir, student, train_loader_plain,
                                         val_loader_plain, writer, wandb, prefix='teacher')

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
            teacher.train()

        for it, (images, _) in tqdm.tqdm(enumerate(train_loader), total=n_batches, desc=" batch", position=1,
                                         leave=False):
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

            writer.add_scalar("train_loss", loss.item(), n_steps)
            writer.add_scalar("epoch", epoch, n_steps)
            writer.add_scalar("teacher_momentum", m, n_steps)
            writer.add_scalar("lr", optim.param_groups[0]['lr'], n_steps)
            writer.add_scalar("weight_decay", optim.param_groups[0]['weight_decay'], n_steps)
            if args.wandb:
                wandb.log({
                    "epoch": epoch,
                    "train_loss": loss.item(),
                    "lr": optim.param_groups[0]['lr'],
                    "weight_decay": optim.param_groups[0]['weight_decay'],
                    "teacher_momentum": m,
                }, step=n_steps)
            n_steps += 1


def eval_model(args, example_viz_img, n_steps, output_dir, model, train_loader_plain, val_loader_plain, writer,
               wandb, prefix):
    current_acc = compute_knn(model.backbone, train_loader_plain, val_loader_plain)
    writer.add_scalar(f'{prefix}_knn_acc', current_acc, n_steps)
    if args.wandb:
        wandb.log({f'{prefix}_knn_acc': current_acc}, step=n_steps)
    if args.visualize:
        if 'vit_' in args.model:
            # TODO how to visualize cls token for mobilevit?
            orig, attentions = dino_attention([model.backbone], args.patch_size, (example_viz_img,),
                                              plot=False,
                                              path=output_dir)
        else:
            orig, attentions = grad_cam(model.backbone, args.model, (example_viz_img,), plot=False,
                                        path=output_dir)

        tsne_fig = t_sne(model.backbone, val_loader_plain, plot=False, path=output_dir)
        if args.wandb:
            wandb.log({f'{prefix}_grads': [wandb.Image(img) for img in attentions]}, step=n_steps)
            wandb.log({f'{prefix}_tsne': wandb.Image(tsne_fig)}, step=n_steps)
    return current_acc


if __name__ == '__main__':
    main(get_args())
