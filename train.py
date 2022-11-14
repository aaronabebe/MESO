import argparse
import os

import pytorch_lightning as pl
import timm.scheduler
import torch
from torch.nn import functional as F

from data import get_dataloader
from models import get_model, get_eval_model
from utils import TENSORBOARD_LOG_DIR, get_experiment_name, get_latest_model_path
from visualize import dino_attention, grad_cam, loss_landscape
from torchmetrics.functional import accuracy


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run training for different ML experiments')
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use.")
    parser.add_argument("--model", type=str, default='resnet_cifar10', help="Model to use.")
    parser.add_argument("--ckpt_path", type=str, help="Override for default model loading dir when loading a model.")

    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam', 'adamw'],
                        help="Optimizer to use.")
    parser.add_argument("--sam", action='store_true', default=False,
                        help='Use SAM in conjunction with standard chosen optimizer.')
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum when training with SGD as optimizer.")
    parser.add_argument("--scheduler", type=str, default='cosine', help="Learning rate decay for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--lr_decay", type=float, default=0.5, help="Learning rate decay for optimizer")
    parser.add_argument("--warmup_steps", type=float, default=3, help="Warmup steps when using cosine LR scheduler.")

    parser.add_argument("--device", type=str, default='cuda', help="Device to use.")  # mps = mac m1 device
    parser.add_argument("--eval", action='store_true', default=False, help='Evaluate model.')
    parser.add_argument("--visualize", type=str, choices=['dino', 'grad_cam', 'landscape'],
                        help="Visualize the loss landscape of the model.")
    return parser.parse_args()


class Net(pl.LightningModule):
    def __init__(self, args: argparse.Namespace):
        super(Net, self).__init__()
        self.args = args
        self.model = get_model(args.model)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y, y_hat, task='multiclass', num_classes=10, top_k=1)

        self.log('train_loss', loss)
        self.log("train_acc", acc)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        y_hat = torch.argmax(y_hat, dim=1)
        acc = accuracy(y, y_hat, task='multiclass', num_classes=10, top_k=1)

        self.log(f'{stage}_loss', loss)
        self.log(f'{stage}_acc', acc)

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, stage='test')

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, stage='val')

    def configure_optimizers(self):
        if self.args.sam:
            raise NotImplementedError('SAM is not working yet!')

        if self.args.optimizer == 'adam':
            optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            optim = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                    weight_decay=self.args.weight_decay, momentum=self.args.momentum)
        elif self.args.optimizer == 'adamw':
            optim = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate,
                                      weight_decay=self.args.weight_decay)
        else:
            raise Exception('Please specify valid optimizer!')

        if self.args.scheduler == 'cosine':
            scheduler = timm.scheduler.CosineLRScheduler(
                optimizer=optim, t_initial=self.args.learning_rate,
                warmup_t=self.args.warmup_steps, decay_rate=self.args.lr_decay
            )

        return [optim], [{"scheduler": scheduler, "interval": "epoch"}]

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        # timm scheduler needs epoch
        scheduler.step(self.current_epoch)


def main(args: argparse.Namespace):
    if args.visualize:
        print(f'Visualizing {args.visualize} for {args.model} model...')

        model = get_eval_model(args.model, ckpt_path=args.ckpt_path)

        dl = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)
        data = next(iter(dl))

        if args.visualize == 'dino':
            dino_attention(model, args.model, data)
        elif args.visualize == 'grad_cam':
            grad_cam(model, args.model, data)
        elif args.visualize == 'landscape':
            loss_landscape(model, args.model, data)
        else:
            print('Please provide a type of visualization you want to do!')
        return

    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
    experiment_name = get_experiment_name(args)
    logger = pl.loggers.TensorBoardLogger(TENSORBOARD_LOG_DIR, name=experiment_name)
    net = Net(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator=args.device
    )
    train_loader = get_dataloader(args.dataset, train=True, batch_size=args.batch_size)
    val_loader = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)

    if args.eval:
        net.load_from_checkpoint(args.ckpt_path if args.ckpt_path else get_latest_model_path(args.model), args=args)
        trainer.test(net, dataloaders=val_loader)
    else:
        trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)
        trainer.test(net, dataloaders=val_loader)


if __name__ == "__main__":
    main(get_args())
