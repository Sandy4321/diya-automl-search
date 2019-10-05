import os
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from settings import PROJECT_ROOT, DATA_DIR

__all__ = ['mnist']


class MNIST(datasets.MNIST):
    def __init__(self, root, train):
        super().__init__(root, train=train, download=False)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.1307], std=[0.3081])
        ])


def mnist(args):
    root = os.path.join(PROJECT_ROOT, DATA_DIR)
    return {
        'size': (1, 28, 28),
        'num_classes': 10,
        'train': DataLoader(
            MNIST(root, train=True),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=True
        ),
        'test': DataLoader(
            MNIST(root, train=False),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            drop_last=False
        )
    }
