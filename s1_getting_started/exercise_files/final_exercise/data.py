import torch

from torchvision import datasets, transforms

def mnist():
    # exchange with the corrupted mnist dataset
    #train = torch.randn(50000, 784)
    #test = torch.randn(10000, 784)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    train = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=True, transform=transform)
    test = datasets.FashionMNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

    return train, test
