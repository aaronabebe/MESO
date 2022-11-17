import json
import math
import os
import sys

import torch
import tqdm
from torch.utils.data import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

from data import get_dataloader, DinoTransforms, default_cifar10_transforms
from models import get_model
from test import compute_embeddings, compute_knn
from utils import get_args, TENSORBOARD_LOG_DIR, get_experiment_name, MultiCropWrapper, MLPHead, DINOLoss, \
    clip_gradients


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
    val_loader_plain_subset = get_dataloader(
        args.dataset, transforms=default_cifar10_transforms(),
        train=False,
        batch_size=args.batch_size,
        sampler=SubsetRandomSampler(list(range(0, len(val_loader_plain), 50)))
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

    lr = args.learning_rate * args.batch_size / 256
    if args.optimizer == 'adam':
        optim = torch.optim.Adam(student.parameters(), lr=lr)
    elif args.optimizer == 'sgd':
        optim = torch.optim.SGD(student.parameters(), lr=lr, weight_decay=args.weight_decay, momentum=args.momentum)
    elif args.optimizer == 'adamw':
        optim = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f'Unknown optimizer: {args.optimizer}')

    # todo: lr/weight decay/momentum scheduler

    # cifar10 trainset contains 50000 imagefixed error in dataloaders
    n_batches = 50000 // args.batch_size
    best_acc = 0
    n_steps = 0

    for epoch in range(args.epochs):
        for i, (images, _) in tqdm.tqdm(enumerate(train_loader), total=n_batches):
            if n_steps % 2000 == 0:
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
                    torch.save(student, f'{TENSORBOARD_LOG_DIR}/dino/{experiment_name}/best.pth')
                    best_acc = current_acc
                student.train()

            images = [img.to(device) for img in images]

            teacher_outputs = teacher(images[:2])
            student_outputs = student(images)

            loss = dino_loss(student_outputs, teacher_outputs, epoch)

            if not math.isfinite(loss.item()):
                print(f"Loss is {loss.item()}, stopping training")
                sys.exit(1)

            optim.zero_grad()
            loss.backward()
            # todo: cancel gradients last layer?
            clip_gradients(student, 2.)
            optim.step()

            with torch.no_grad():
                m = args.momentum_teacher
                for param_q, param_k in zip(student.parameters(), teacher.parameters()):
                    param_k.data.mul(m).add_((1 - m) * param_q.detach().data)

            writer.add_scalar("train_loss", loss.item(), n_steps)
            n_steps += 1


if __name__ == '__main__':
    main(get_args())
