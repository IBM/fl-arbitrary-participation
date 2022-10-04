import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import FashionMNIST
from PIL import Image
from typing import Any, Callable, Optional, Tuple

# These enhanced classes do transformation only once and save the data in memory, to speed up the overall computation

class FashionMNISTEnhanced(FashionMNIST):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device: torch.device = None,
    ) -> None:
        super(FashionMNISTEnhanced, self).__init__(root, train, transform, target_transform, download)

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img.numpy(), mode='L')

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.stack(self.targets_transformed)

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


class CIFAR10Enhanced(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
            device : torch.device = None,
    ) -> None:
        super(CIFAR10Enhanced, self).__init__(root, train, transform, target_transform, download)

        self.data_transformed = []
        self.targets_transformed = []

        for img in self.data:
            img = Image.fromarray(img)

            if self.transform is not None:
                img = self.transform(img)

            self.data_transformed.append(img)

        for target in self.targets:
            if self.target_transform is not None:
                target = self.target_transform(target)

            self.targets_transformed.append(target)

        self.data_transformed = torch.stack(self.data_transformed)
        self.targets_transformed = torch.tensor(self.targets_transformed, dtype=torch.int64)  # Note: this is different from the MNIST class

        if device is not None:
            self.data_transformed = self.data_transformed.to(device)
            self.targets_transformed = self.targets_transformed.to(device)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data_transformed[index], self.targets_transformed[index]
        return img, target


def load_data(dataset, data_path, device):
    if dataset == 'FashionMNIST':
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        data_train = FashionMNISTEnhanced(data_path,
                                  transform=transform_normalize,
                                  download=True, device=device)  # download=True for the first time
        data_test = FashionMNISTEnhanced(data_path,
                                 train=False,
                                 transform=transform_normalize, device=device)

    elif dataset == 'CIFAR10':
        transform_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        data_train = CIFAR10Enhanced(data_path,
                             transform=transform_normalize,
                             download=True, device=device)  # download=True for the first time
        data_test = CIFAR10Enhanced(data_path,
                            train=False,
                            transform=transform_normalize, device=device)
    else:
        raise Exception('Unknown dataset name.')

    return data_train, data_test
