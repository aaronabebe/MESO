import argparse

import pytorch_lightning as pl
import torch
from torch.nn import functional as F

from data import get_dataloader
from models import get_model
from utils import get_experiment_name
from visualize import dino_attention


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run training for different ML experiments')
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--model", type=str, default='vit_tiny_cifar10', help="Model to use.")
    parser.add_argument("--device", type=str, default='cpu', help="Device to use.")  # mps = mac m1 device
    parser.add_argument("--visualize", type=str, choices=['2D', '3D'],
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
        acc = torch.eq(y_hat.argmax(-1), y).float().mean()

        self.log('train_loss', loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = torch.eq(y_hat.argmax(-1), y).float().mean()

        self.log('val_loss', loss)
        self.log('val_acc', acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)


def main(args: argparse.Namespace):
    if args.visualize:
        # TODO fix model loading and move viz to different file
        print(f'Visualizing the loss landscape in {args.visualize} of the {args.model} model.')
        name = get_experiment_name(args)

        print('Loading model from checkpoint:', name)
        model = get_model(args.model)
        ckpt = torch.load('./tb_logs/vit_tiny_cifar10_30_128/version_1/checkpoints/epoch=5-step=2346.ckpt')
        model.load_state_dict({k[len('model.'):]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')})
        for p in model.parameters():
            p.requires_grad = False
        model.eval()

        dl = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)
        data = next(iter(dl))
        # data = torch.rand(1, 1, 3, 32, 32)
        # loss_landscape(model, data)
        # grad_cam(model, args.model, data)
        dino_attention(model, args.model, data)
        return

    experiment_name = get_experiment_name(args)

    # TODO lightning checkpoint callback
    logger = pl.loggers.TensorBoardLogger('tb_logs', name=experiment_name)
    net = Net(args)
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        accelerator=args.device,
        default_root_dir='./models'
    )
    train_loader = get_dataloader(args.dataset, train=True, batch_size=args.batch_size)
    val_loader = get_dataloader(args.dataset, train=False, batch_size=args.batch_size)
    trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main(get_args())
