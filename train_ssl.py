import json
import math
import os
import sys

import torch
import tqdm
from torch.utils.data import RandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader, DinoTransforms, default_cifar10_transforms
from dino_utils import MultiCropWrapper, MLPHead, DINOLoss, clip_gradients
from dino_utils import get_params_groups, dino_cosine_scheduler, cancel_gradients_last_layer
from models import get_model
from test import compute_embeddings, compute_knn
from utils import get_args, TENSORBOARD_LOG_DIR, get_experiment_name


def main(args):
    print(vars(args))
    device = torch.device(args.device)

    transforms = DinoTransforms(args.input_size, args.n_local_crops, args.local_crops_scale,
                                args.global_crops_scale)
    train_loader = get_dataloader(args.dataset, transforms=transforms, train=True,
                                  batch_size=args.batch_size)
    train_loader_plain = get_dataloader(args.dataset, transforms=default_cifar10_transforms(), train=True,
                                        batch_size=args.batch_size)
    val_loader_plain = get_dataloader(args.dataset, transforms=default_cifar10_transforms(), train=False,
                                      batch_size=args.batch_size)

    # sample one random batch for embedding visualization
    val_loader_plain_subset = get_dataloader(
        args.dataset, transforms=default_cifar10_transforms(),
        train=False,
        batch_size=args.batch_size,
        sampler=RandomSampler(val_loader_plain.dataset, replacement=True, num_samples=args.batch_size)
    )

    os.makedirs(f'{TENSORBOARD_LOG_DIR}/dino/', exist_ok=True)
    experiment_name = get_experiment_name(args)
    writer = SummaryWriter(f'{TENSORBOARD_LOG_DIR}/dino/{experiment_name}')
    writer.add_text("args", json.dumps(vars(args)))

    student = get_model(args.model)
    student = MultiCropWrapper(student, MLPHead(in_dim=192, out_dim=args.out_dim))
    student = student.to(device)

    teacher = get_model(args.model)
    teacher = MultiCropWrapper(teacher, MLPHead(in_dim=192, out_dim=args.out_dim))
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

    # cifar10 trainset contains 50000 imagefixed error in dataloaders
    n_batches = 50000 // args.batch_size
    best_acc = 0
    n_steps = 0

    for epoch in range(args.epochs):
        for it, (images, _) in tqdm.tqdm(enumerate(train_loader), total=n_batches):
            if n_steps > 1 and n_steps % 1000 == 0:
                print('Evaluating on validation set...')
                student.eval()
                embs, imgs, labels = compute_embeddings(student.backbone, val_loader_plain_subset)
                writer.add_embedding(
                    embs,
                    metadata=labels,
                    label_img=imgs,
                    global_step=n_steps,
                    tag="embedding"
                )

                # knn eval
                current_acc = compute_knn(student.backbone, train_loader_plain, val_loader_plain)
                writer.add_scalar('knn_acc', current_acc, n_steps)
                if current_acc > best_acc:
                    save_dict = {
                        'student': student.state_dict(),
                        'teacher': teacher.state_dict(),
                        'optimizer': optim.state_dict(),
                        'epoch': epoch + 1,
                        'args': args,
                        'dino_loss': dino_loss.state_dict(),
                    }
                    torch.save(save_dict, f'{TENSORBOARD_LOG_DIR}/dino/{experiment_name}/best.pth')
                    best_acc = current_acc
                student.train()

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
            n_steps += 1


if __name__ == '__main__':
    main(get_args())
