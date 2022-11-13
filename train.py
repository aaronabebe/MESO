import argparse
import os

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from data import get_dataloader
from models import get_model, get_eval_model
from utils import accuracy, TENSORBOARD_LOG_DIR
from visualize import dino_attention, grad_cam, loss_landscape


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run training for different ML experiments')
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use.")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate.")
    parser.add_argument("--model", type=str, default='vit_tiny_cifar10', help="Model to use.")
    parser.add_argument("--optimizer", type=str, default='sgd', choices=['sgd', 'adam'], help="Optimizer to use.")
    parser.add_argument("--weight_decay", type=float, default=0.0005, help="Weight decay for optimizer")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use.")  # mps = mac m1 device
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
        acc = accuracy(y, y_hat)

        self.log('train_loss', loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y, y_hat)

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        if self.args.optimizer == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate,
                                   weight_decay=self.args.weight_decay)
        else:
            raise Exception('Please specify valid optimizer!')


def main(args: argparse.Namespace):
    if args.visualize:
        print(f'Visualizing {args.visualize} for {args.model} model...')

        model = get_eval_model(args.model)

        # data = torch.rand(1, 1, 3, 32, 32)
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
    logger = pl.loggers.TensorBoardLogger(TENSORBOARD_LOG_DIR, name=args.model)
    net = Net(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator=args.device
    )
    train_loader = get_dataloader(args.dataset, train=True, batch_size=args.batch_size)
    val_loader = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main(get_args())
