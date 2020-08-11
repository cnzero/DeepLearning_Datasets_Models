from torchvision import datasets, transforms
from base import BaseDataLoader
import numpy as np


class MNIST_DataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class Fashion_DataLoader(BaseDataLoader):
    # Fashion_MNIST dataset
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # data/Fashion_MNIST/fashion-mnist_train.csv
        # data/Fashion_MNIST/fashion-mnist_test.csv


        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
