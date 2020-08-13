from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.mnist import FashionMNIST
from base import BaseDataLoader
import numpy as np
import pandas as pd


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


class Fashion_Dataset(Dataset):
    def __init__(self, data_dir, train=True, transform=None):
        self.data_dir = data_dir
        self.train = train
        if self.train: # for train data file
            data = pd.read_csv(self.data_dir+'fashion-mnist_train.csv')
        else:          # for tests data file
            data = pd.read_csv(self.data_dir+'fashion-mnist_test.csv')
        
        self.fashion = list(data.values)
        ### --- debug
        print('debug', len(self.fashion), len(self.fashion[0]))
        self.transform = transform

        label, image = [], []

        for sample in self.fashion:
            label.append(sample[0])  # first column for fashion labels
            image.append(sample[1:]) # last 784 columns for pixels
        self.labels = np.asarray(label)
        self.images = np.asarray(image).reshape(-1, 28, 28).astype('float32')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = self.images[idx]

        if self.transform is not None:
            pil_image = Image.fromarray(np.uint8(image))
            image = self.transform(pil_image)

        return image, label


class Fashion_DataLoader(BaseDataLoader):
    # Fashion_MNIST dataset
    def __init__(self, data_dir, batch_size=64, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        # data/Fashion_MNIST/fashion-mnist_train.csv
        # data/Fashion_MNIST/fashion-mnist_test.csv
        transform = transforms.Compose([
            transforms.Resize((227, 227)), 
            transforms.ToTensor(),
            transforms.Normalize((0.01307,), (0.3081,))
        ])
        # transform = None
        self.data_dir = data_dir
        self.dataset =Fashion_Dataset(self.data_dir, train=training, transform=transform)

        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
