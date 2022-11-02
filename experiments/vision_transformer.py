import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def train(args):
    batch_size = args['batch_size']
    trainset = CIFAR10(root='./data', train=True, download=True, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    print(images[0].shape)

    plt.imshow(np.transpose(images[0].numpy(), (1, 2, 0)))
    plt.show()


def main():
    args = {
        'batch_size': 1,
    }
    train(args)


if __name__ == "__main__":
    main()
