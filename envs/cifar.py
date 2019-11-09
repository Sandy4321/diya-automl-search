import os
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from settings import PROJECT_ROOT, DATA_DIR

__all__ = ['cifar10']


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train):
        super().__init__(root, train=train, download=False)
        if train:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])

        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])


def cifar10(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    num_train = 50000
    indices = list(range(num_train))
    split = int(np.floor(args.split_ratio*num_train))
    return {
        'size': (3, 32, 32),
        'num_classes': 10,
        'train': DataLoader(
            CIFAR10(root, train=True),
            sampler=SubsetRandomSampler(indices[:split]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
        ),
        'val': DataLoader(
            CIFAR10(root, train=True),
            sampler=SubsetRandomSampler(indices[split:]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True,
        ),
        'test': DataLoader(
            CIFAR10(root, train=False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False
        )
    }
