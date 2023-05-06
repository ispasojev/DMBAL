import random

import numpy as np
import torch
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

from src.settings import DATA_ROOT


def get_dataset(name, path=DATA_ROOT):
    if name == 'CIFAR10':
        return get_CIFAR10(path)


def get_CIFAR10(path):
    data_tr = datasets.CIFAR10(path, train=True, download=True)
    data_te = datasets.CIFAR10(path, train=False, download=True)
    X_tr = data_tr.data
    Y_tr = torch.from_numpy(np.array(data_tr.targets))
    X_te = data_te.data
    Y_te = torch.from_numpy(np.array(data_te.targets))

    return X_tr, Y_tr, X_te, Y_te


class DataHandler(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.X[index], self.Y[index]
        if self.transform is not None:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y, index

    def __len__(self):
        return len(self.X)


def get_handler_and_args(name):
    if name == 'CIFAR10':
        return DataHandler, {
            'dims': [32, 32, 3],
            'transform': transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))]),
            'loader_tr_args': {'batch_size': 128, 'num_workers': 1},
            'loader_te_args': {'batch_size': 1000, 'num_workers': 1},
            'transformTest': transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
        }


