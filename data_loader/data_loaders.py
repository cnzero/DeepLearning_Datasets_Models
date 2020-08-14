from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.mnist import FashionMNIST
from base import BaseDataLoader
import numpy as np
import pandas as pd
import os

# for NinaPro
from scipy import io


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



class NinaPro_Dataset(Dataset):
    def __init__(self, NinaProPath='/home/Disk2T-1/NinaPro/', promoteInfo=True,
                 DB = 1,
                 repetition=[0,1,2,3,4,5,6,8,9], 
                 classes, 
                 subject=1,
                 emg_channels=[0,1,2,3],
                 normalize=None,
                 transform=None,
                 preprocessing_functions={'low_pass_filter':'params', 'high_pass_filter':'params'},
                 augmentation_functions= {'rotate':'params'},
                 size_factor=1):
        
        # print promotion informations
        if promoteInfo:
            pass

        ###---asserting for input parameters
        #-NinaProPath
        assert os.path.exists(NinaProPath),\
            print('Something path wrong.')
        self.NinaProPath = NinaProPath
        #-DB
        assert isinstance(DB, int),\
            print('DB should be a int value.')
        self.DB = DB
        #-repetition
        assert isinstance(repetition, list) and set(repetition).issubset(list(range(10+1))),\
            print('repetition should be list type and is subset of [0,1,2,3,4,5,6,7,8,9,10].')
        self.repetition = repetition
        #-subject
        assert isinstance(subject, int),\
            print('One No. subject with int type.')
        if self.DB==1:
            assert 1<=subject<=27,\
                print('subject No. should be in range of [1, 27]')
        elif self.DB==2:
            assert 1<=subject<=40,\
                print('subject No. should be in range of [1, 40]')
        elif 3<=self.DB<=6:
            assert 1<=subject<=10,\
                print('subject No. should be in range of [1, 10]')
        elif self.DB==7:
            assert 1<=subject<=22,\
                print('subject No. should be in range of [1, 22]')
        self.subject = subject
        #-emg_channels
        assert isinstance(emg_channels, list),\
            print('emg_channels should be list type for channels selection.')
        self.emg_channels=emg_channels

        #-read key values from .mat files from different [DB] and different [subject]
        if self.DB==1:
            # Ej, j=1,2,3, from three Experiments to construct a whole dataset
            # example
            # '/home/Disk2T-1/NinaPro/DB1mat/Si_A1_Ej.mat', i=1,2,...,27, j=1,2,3
            for j in range(1, 3+1): 
                matPath = self.NinaProPath+'DB'+str(self.DB)+'mat/'+'S'+str(self.subject)+'_A1_E'+str(j)+'.mat'
                mat = io.loadmat(matPath)
                # get data from mat dictionary
                if j==1:
                    # - basic information of this experiment or subject
                    #-personal
                    self.gender = mat['gender'] if 'gender' in mat.keys() else None
                    self.age = mat['age'] if 'age' in mat.keys() else None
                    self.weight = mat['weight'] if 'weight' in mat.keys() else None
                    self.height = mat['height'] if 'height' in mat.keys() else None
                    self.circumference = mat['circumference'] if 'circumference' in mat.keys() else None
                    self.laterality = mat['laterality'] if 'laterality' in mat.keys() else None
                    # acquisition systems
                    self.sensor = mat['sensor'] if 'sensor' in mat.keys() else None
                    self.frequency = mat['frequency'] if 'frequency' in mat.keys() else None
                    self.exercise = mat['exercise'] if 'exercise' in mat.keys() else None
                    self.time = mat['time'] if 'time' in mat.keys() else None
                    self.daytesting = mat['daytesting'] if 'daytesting' in mat.keys() else None
                    # - input
                    assert 'emg' in mat.keys(), \
                        print("Why no 'emg' data in .mat files?")
                    self.emg = mat['emg']
                    self.acc = mat['acc'] if 'acc' in mat.keys() else None
                    self.gyro = mat['gyro'] if 'gyro' in mat.keys() else None
                    self.mag = mat['mag'] if 'mag' in mat.keys() else None
                    self.reobject = mat['reobject'] if 'reobject' in mat.keys() else None
                    self.glove = mat['glove'] if 'glove' in mat.keys() else None
                    self.inclin = mat['inclin'] if 'inclin' in mat.keys() else None

                    # - output
                    self.stimulus = mat['stimulus'] if 'stimulus' in mat.keys() else None
                    self.restimulus = mat['restimulus'] if 'restimulus' in mat.keys() else None
                    self.repetition = mat['repetition'] if 'repetition' in mat.keys() else None
                    self.rerepetition = mat['rerepetition'] if 'rerepetition' in mat.keys() else None
                    self.object = mat['object'] if 'object' in mat.keys() else None
                    self.reobject = mat['reobject'] if 'reobject' in mat.keys() else None

                # j=2,3
                # raw data concat and update gesture labels.

        elif self.DB==2:
            pass
        elif 3<=self.DB<=6:
            pass
        elif self.DB==7:
            pass


    def __getitem__(self, index):
        xs, ys = self.Xs[index], self.Ys[index]
        if self.transform is not None:
            xs = self.transform(xs)
        return xs, ys

    def __len__(self):
        return len(self.Xs)
    