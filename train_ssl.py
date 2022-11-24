import json
import math
import os
import sys
import pprint

import torch

# enable if dataloaders run into "too many open files" error
# torch.multiprocessing.set_sharing_strategy('file_system')

import tqdm
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader, DinoTransforms, get_mean_std, default_transforms
from dino_utils import MultiCropWrapper, MLPHead, DINOLoss, clip_gradients
from dino_utils import get_params_groups, dino_cosine_scheduler, cancel_gradients_last_layer
from models.models import get_model
from test import compute_embeddings, compute_knn
from utils import get_args, TENSORBOARD_LOG_DIR, get_experiment_name
from visualize import dino_attention


def main(args):
    pprint.pprint(vars(args))
    device = torch.device(args.device)

    if args.wandb:
        import wandb
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

    train_loader = get_dataloader(args.dataset, transforms=transforms, train=True,
                                  batch_size=args.batch_size)
    train_loader_plain = get_dataloader(args.dataset, train=True,
                                        batch_size=args.batch_size)
    val_loader_plain = get_dataloader(args.dataset, train=False,
                                      batch_size=args.batch_size)

    # sample one random batch for embedding visualization
    val_loader_plain_subset = get_dataloader(
        args.dataset,
        train=False,
        batch_size=1,
        transforms=default_transforms(480, *get_mean_std(args.dataset)),  # using a larger input size for visualization
        sampler=RandomSampler(val_loader_plain.dataset, replacement=True, num_samples=args.batch_size)
    )

    os.makedirs(f'{TENSORBOARD_LOG_DIR}/dino/', exist_ok=True)
    experiment_name = get_experiment_name(args)
    writer = SummaryWriter(f'{TENSORBOARD_LOG_DIR}/dino/{experiment_name}')
    writer.add_text("args", json.dumps(vars(args)))

    student = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit' in args.model else None,
        img_size=args.input_size if 'vit' in args.model else None
    )
    student = MultiCropWrapper(student, MLPHead(in_dim=args.in_dim, out_dim=args.out_dim))
    student = student.to(device)

    teacher = get_model(
        args.model,
        in_chans=args.input_channels,
        num_classes=args.num_classes,
        patch_size=args.patch_size if 'vit' in args.model else None,
        img_size=args.input_size if 'vit' in args.model else None
    )
    teacher = MultiCropWrapper(teacher, MLPHead(in_dim=args.in_dim, out_dim=args.out_dim))
    teacher = teacher.to(device)

    # teacher gets student weights and doesnt learn
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad = False

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
    print(f"Loss, optimizer and schedulers ready.")

    # cifar10 trainset contains 50000 images
    n_batches = len(train_loader.dataset) // args.batch_size
    best_acc = 0
    n_steps = 0

    for epoch in range(args.epochs):
        print('Evaluating on validation set...')
        teacher.eval()

        # TODO fix this: compute embeddings for tensorboard
        # embs, imgs, labels = compute_embeddings(student.backbone, val_loader_plain_subset)
        # writer.add_embedding(
        #     embs,
        #     metadata=labels,
        #     label_img=imgs,
        #     global_step=n_steps,
        #     tag="embedding"
        # )

        # knn eval
        current_acc = compute_knn(teacher.backbone, train_loader_plain, val_loader_plain)
        writer.add_scalar('knn_acc', current_acc, n_steps)
        if args.wandb:
            wandb.log({'knn_acc': current_acc}, step=n_steps)

        images, labels = next(iter(val_loader_plain_subset))
        images, labels = images.to(device), labels.to(device)
        orig, attentions = dino_attention(teacher.backbone, args.patch_size, (images, labels), plot=False)
        if args.wandb:
            wandb.log({'orig': wandb.Image(orig)}, step=n_steps)
            wandb.log({'attention_maps': [wandb.Image(img) for img in attentions]}, step=n_steps)

        if current_acc > best_acc:
            save_dict = {
                'student': student.state_dict(),
                'teacher': teacher.state_dict(),
                'optimizer': optim.state_dict(),
                'epoch': epoch + 1,
                'args': args,
                'dino_loss': dino_loss.state_dict(),
            }
            save_path = f'{TENSORBOARD_LOG_DIR}/dino/{experiment_name}/best.pth'
            torch.save(save_dict, save_path)
            if args.wandb:
                artifact = wandb.Artifact('model', type='model')
                artifact.add_file(save_path)
                wandb.log_artifact(artifact)

            best_acc = current_acc
        teacher.train()

        for it, (images, _) in tqdm.tqdm(enumerate(train_loader), total=n_batches):
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
            writer.add_scalar("lr", optim.param_groups[0]['lr'], n_steps)
            writer.add_scalar("weight_decay", optim.param_groups[0]['weight_decay'], n_steps)
            if args.wandb:
                wandb.log({
                    "train_loss": loss.item(),
                    "lr": optim.param_groups[0]['lr'],
                    "weight_decay": optim.param_groups[0]['weight_decay'],
                }, step=n_steps)
            n_steps += 1


if __name__ == '__main__':
    main(get_args())
