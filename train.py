import argparse
import os

import pytorch_lightning as pl

from data import get_dataloader, default_cifar10_transforms, default_transforms
from models.models import get_eval_model, LitNet
from utils import TENSORBOARD_LOG_DIR, get_experiment_name, get_args


def main(args: argparse.Namespace):
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    experiment_name = get_experiment_name(args)
    logger = pl.loggers.TensorBoardLogger(TENSORBOARD_LOG_DIR, name=experiment_name)
    net = LitNet(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator=args.device
    )
    train_loader = get_dataloader(args.dataset, transforms=default_cifar10_transforms, train=True,
                                  batch_size=args.batch_size)
    val_loader = get_dataloader(args.dataset, transforms=default_transforms, train=False, batch_size=args.batch_size)

    if args.eval:
        net.model = get_eval_model(args.model, args.ckpt_path)
        trainer.test(net, dataloaders=val_loader)
    else:
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(net, dataloaders=val_loader)


if __name__ == "__main__":
    main(get_args())
