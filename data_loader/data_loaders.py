from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.datasets.mnist import FashionMNIST
from torchvision import transforms
from base import BaseDataLoader
import numpy as np
import pandas as pd
import os

# for NinaPro
from scipy import io
from data_loader.ninapro_preprocessing import low_pass_filter, high_pass_filter
from data_loader.ninapro_augmentation import jitter, scale, rotate, mag_warp, time_warp, permute
from data_loader.ts_features import get_tsfresh_features

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
        # print('debug', len(self.fashion), len(self.fashion[0]))
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


def InputOutput_instantaneous(Xs, Ys, params):
    # parameters are set here, rather than using transfering parameters
    Xs = np.reshape(Xs, (Xs.shape[0], 1, Xs.shape[1]))
    Ys = Ys.squeeze(1)
    return Xs, Ys

def InputOutput_rawImages(Xs, Ys, params={'LW':30, 'LI':10}):
    LW = params['LW']
    LI = params['LI']

    N_channels = Xs.shape[1]
    xs = np.zeros((1, 1, LW, N_channels))
    ys = np.zeros((1, 1))
    # same gesture data sliding window splition
    for label in np.unique(Ys):
        rawData_label = Xs[Ys.squeeze(1)==label, :]
        Length = rawData_label.shape[0]
        Nw = int((Length-LW)/LI + 1)
        for w in range(Nw):
            startIndex = w*LI
            endIndex = startIndex + LW
            x_window = Xs[startIndex:endIndex, :]
            y_window = label * np.ones((1,1), dtype=int)

            # numpy Xs/Ys concatenate
            xs = np.concatenate( (xs, x_window.reshape(1,1,LW, N_channels) ), axis=0 )
            ys = np.concatenate( (ys, y_window), axis=0 )

    return xs[1:, :, :], ys[1:, :].squeeze(1)

def InputOutput_TCNseq(Xs, Ys, params={'LW':200, 'LI':100}):
    LW = params['LW']
    LI = params['LI']

    N_channels = Xs.shape[1]
    xs = np.zeros((1, LW, N_channels))
    ys = np.zeros((1, 1))
    # same gesture data sliding window splition
    for label in np.unique(Ys):
        rawData_label = Xs[Ys.squeeze(1)==label, :]
        Length = rawData_label.shape[0]
        Nw = int((Length-LW)/LI + 1)
        for w in range(Nw):
            startIndex = w*LI
            endIndex = startIndex + LW
            x_window = Xs[startIndex:endIndex, :]
            y_window = label * np.ones((1,1), dtype=int)

            # numpy Xs/Ys concatenate
            xs = np.concatenate( (xs, x_window.reshape(1,LW, N_channels) ), axis=0 )
            ys = np.concatenate( (ys, y_window), axis=0 )

    return xs[1:, :, :], ys[1:, :].squeeze(1)

def InputOutput_features(Xs, Ys, params):
    print('InputOutput params', params)
    # input
    #       [self.Xs], N x channels
    #       [self.Ys], (N,) labels
    # output
    #       [self.Xs], m_samples x n_features
    #       {self.Ys}, (m_samples, )
    LW = params['LW']
    LI = params['LI']
    features = params['features']
    # features = {
    #             'absolute_sum_of_changes': None, 
    #             'cid_ce': [{'normalize': False}, {'normalize': True}],
    #             'kurtosis': None, 
    #             'large_standard_deviation': [{'r': 0.2}, {'r': 0.8}],
    #             'linear_trend': [{'attr': 'stderr'}], 
    #             'maximum': None, 
    #             'mean_abs_change': None, 
    #             'minimum': None, 
    #             'percentage_of_reoccurring_datapoints_to_all_datapoints': None, 
    #             'percentage_of_reoccurring_values_to_all_values': None, 
    #             'quantile': [{'q': 0.1}, {'q': 0.6}, {'q': 0.9}], 
    #             'quantile': [{'q': 0.2}, {'q': 0.6}, {'q': 0.9}], 
    #             'ratio_value_number_to_time_series_length': None, 
    #             'sample_entropy': None, 
    #             'standard_deviation': None, 
    #             'sum_of_reoccurring_data_points': None, 
    #             'sum_of_reoccurring_values': None, 
    #             'variance': None, 
    #             'variation_coefficient': None
    #            }
    # features extraction
    N_channels = Xs.shape[1]

    for i,label in enumerate(np.unique(Ys).reshape((-1,1)).tolist()):
        # first to init Xs dimentions, which are heavily dependent on the features
        conditions_label = Ys==label[0]
        conditions_label = conditions_label.squeeze(1)
        X_data = Xs[conditions_label, :]
        # print(i, label, X_data.shape)
        if i==0:
            # Xs_tsfresh = get_tsfresh_features(X=X_data, LW=LW, LI=LI, features_Parameters=features)
            Xs_tsfresh = get_tsfresh_features(X=X_data, LW=LW, LI=LI, features=features)
            Ys_tsfresh = label * np.ones(shape=(Xs_tsfresh.shape[0], 1), dtype=int)
        else:
            # xs = get_tsfresh_features(X=X_data, LW=LW, LI=LI, features_Parameters=features)
            xs = get_tsfresh_features(X=X_data, LW=LW, LI=LI, features=features)
            ys = label * np.ones(shape=(xs.shape[0], 1), dtype=int)
            # concentate
            Xs_tsfresh = np.concatenate( (Xs_tsfresh, xs), axis=0)
            Ys_tsfresh = np.concatenate( (Ys_tsfresh, ys), axis=0)

        # print('Xs shape: ', Xs.shape,'X_data shape: ', X_data.shape)

    return Xs_tsfresh, Ys_tsfresh

 

class NinaPro_Dataset(Dataset):
    def __init__(self, data_dir='/home/Disk2T-1/NinaPro/', promoteInfo=True,
                 DB = 1,
                 train=True,
                 trainRepeatIndex=[0,1,3,4,6,8,9], 
                 classes=list(range(1, 53)), 
                 subject=1,
                 emg_channels=[0,1,2,3],
                 normalize=None,
                 transform=None,
                 preprocessing_functions={ \
                                        #   'low_pass_filter': {'f':1, 'fs':100}, \
                                        #   'high_pass_filter':{'f':3, 'fs':100}, \
                                         },
                 augmentation_functions= {\
                                        #   'scale':{'sigma':0.2},\
                                        #   'rotate':{'rotation':2, 'mask':None}, \
                                         },
                #  InputOutput_function='InputOutput_instantaneous',
                 InputOutput_function='InputOutput_rawImages', 
                 InputOutput_params={'LW':30, 'LI':10},
                #  InputOutput_function='InputOutput_TCNseq',
                #  InputOutput_function=InputOutput_features,
                #  InputOutput_params = {'LW':200, 'LI':100, \
                #                        'features':{'maximum':None, \
                #                                    'minimum':None, \
                #                                    'variance':None,
                #                                    }
                #                        },
                 size_factor=0):
        
        # print promotion informations
        if promoteInfo:
            pass

        ###---asserting for input parameters
        #-data_dir
        assert os.path.exists(data_dir),\
            print('Something path wrong.')
        self.data_dir = data_dir
        #-DB
        assert isinstance(DB, int),\
            print('DB should be a int value.')
        self.DB = DB
        #-train or test dataset return
        self.train = train
        #-repetition
        assert isinstance(trainRepeatIndex, list) and set(trainRepeatIndex).issubset(list(range(10+1))),\
            print('repetition should be list type and is subset of [0,1,2,3,4,5,6,7,8,9,10].')
        self.trainRepeatIndex = trainRepeatIndex
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

        self.normalize = normalize
        self.transform = transform
        self.preprocessing_functions = preprocessing_functions
        self.augmentation_functions = augmentation_functions
        self.size_factor = size_factor


        #-read key values from .mat files from different [DB] and different [subject]
        #---------------!!!Attention!!!-------DB6,DB7, are not considered. 
        DB_Ej = {1:3, 2:3, 3:3, 4:3, 5:3}
        for j in range(1, DB_Ej[self.DB]+1):
            if self.DB==1:
                matPath = self.data_dir+'DB'+str(self.DB)+'mat/'+'S'+str(self.subject)+'_A1_E'+str(j)+'.mat'
            elif 2<=self.DB<=5:
                matPath = self.data_dir+'DB'+str(self.DB)+'mat/'+'S'+str(self.subject)+'_E'+str(j)+'_A1.mat'
            elif self.DB==6:
                pass
            elif self.DB==7:
                pass
            elif self.DB==8:
                pass
            elif self.DB==9:
                pass

            # read data of dictionary from .mat files.
            mat = io.loadmat(matPath)
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

                # - output
                self.repetition = mat['repetition'] if 'repetition' in mat.keys() else None
                self.rerepetition = mat['rerepetition'] if 'rerepetition' in mat.keys() else None
                self.stimulus = mat['stimulus'] if 'stimulus' in mat.keys() else None
                self.restimulus = mat['restimulus'] if 'restimulus' in mat.keys() else None
                self.glove = mat['glove'] if 'glove' in mat.keys() else None
                self.inclin = mat['inclin'] if 'inclin' in mat.keys() else None
                self.object = mat['object'] if 'object' in mat.keys() else None
                self.reobject = mat['reobject'] if 'reobject' in mat.keys() else None

            # j=2,3
            # raw data concat and update gesture labels.
            else:  # j=2,3, for another experimental data
                # input
                assert 'emg' in mat.keys(),\
                    print("Why no 'emg' data in .mat files?")
                self.emg = np.vstack((self.emg, mat['emg']))
                if self.acc is not None:
                    self.acc = np.vstack((self.acc, mat['acc']))
                if self.gyro is not None:
                    self.gyro = np.vstack(self.gyro, mat['gyro'])
                if self.mag is not None:
                    self.mag = np.vstack(self.mag, mat['mag'])

                # output, 
                self.repetition = np.vstack((self.repetition, mat['repetition']))
                self.rerepetition = np.vstack((self.rerepetition, mat['rerepetition']))
                # -- update gesture labels
                lastGestureNums = len(np.unique(self.stimulus))-1
                stimulus = mat['stimulus']
                stimulus = stimulus + (stimulus!=0)*lastGestureNums
                self.stimulus = np.vstack((self.stimulus, stimulus))


                restimulus = mat['restimulus']
                restimulus = restimulus + (restimulus!=0)*lastGestureNums
                self.restimulus = np.vstack((self.restimulus, restimulus))
                if self.glove is not None and 'glove' in mat.keys():
                    self.glove = np.vstack((self.glove, mat['glove']))
                if self.inclin is not None and 'inclin' in mat.keys():
                    self.inclin = np.vstack((self.inclin, mat['inclin']))
                if self.object is not None and 'object' in mat.keys():
                    self.object = np.vstack((self.object, mat['object']))
                if self.reobject is not None and 'reobject' in mat.keys():
                    self.reobject = np.vstack((self.reobject, mat['reobject']))

        # raw signal preprocessing before any further split and extraction
        if self.preprocessing_functions is not None:
            for func, params in zip(self.preprocessing_functions.keys(), self.preprocessing_functions.values()):
                self.emg =   eval(func)(self.emg, params)
        # split for different train-test datasets based on [self.trainRepeatIndex]
        conditionRepeatIndex = np.empty_like(self.rerepetition)
        if self.train:
            x = self.emg
            y = self.restimulus
            r = self.rerepetition
            # raw sEMG signal data augmentation for train datasets
            for _ in range(self.size_factor):
                # print('DEBUG: ', self.emg.shape)
                for func, params in zip(self.augmentation_functions.keys(), self.augmentation_functions.values()):
                    self.emg = np.concatenate((self.emg, eval(func)(x, params) ), axis=0)
                    self.restimulus = np.concatenate((self.restimulus, y), axis=0)
                    self.rerepetition = np.concatenate((self.rerepetition, r), axis=0)
            assert self.emg.shape[0]==self.restimulus.shape[0]==self.rerepetition.shape[0], \
                print('augmentation concatenate wrong!!!')
            conditionRepeatIndex = np.any(self.rerepetition==self.trainRepeatIndex, axis=1)
        else:
            testRepetitionIndex = []
            for i in range(10+1):
                if i not in self.trainRepeatIndex:
                    testRepetitionIndex.append(i)
            conditionRepeatIndex = np.any(self.rerepetition==testRepetitionIndex, axis=1)
        conditionRepeatIndex = conditionRepeatIndex.reshape((-1, 1)) 
        
        conditionNotRest = self.restimulus!=0
        conditions = conditionRepeatIndex * conditionNotRest
        self.Xs =        self.emg[conditions[:, 0], :]    # N x N_channels
        self.Ys = self.restimulus[conditions[:, 0], :] -1 # (N, )
        #TODO
        ### label squeeze and selection
        ### re-arrange labels or one-hot encoding

        # how to feed for the Models
        self.Xs, self.Ys = eval(InputOutput_function)(self.Xs, self.Ys, InputOutput_params)


    def __getitem__(self, index):
        xs, ys = self.Xs[index, :], self.Ys[index]
        if self.transform is not None:
            xs = self.transform(xs)
        # return torch.from_numpy(xs), torch.from_numpy(ys)
        return xs, ys

    def __len__(self):
        return len(self.Xs)




   
class NinaPro_DataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size=64, shuffle=True, validation_split=0.1, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = NinaPro_Dataset()
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
