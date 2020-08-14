import os
import scipy.io
import numpy as np
from ninapro_preprocessing import *
from ninapro_augmentation import *
from torch.utils.data import Dataset


class NinaproDataset(Dataset):
    '''Create dataset from Ninapro DB1'''

    def __init__(self, repetitions, classes, subject=1, input_dir='./dataset/Ninapro-DB1-Proc', data_type='raw',
                 emg_channels=10, window_size_H=15, window_size_W=10, window_step=15, drop_last=True, normalize=None,
                 transform=None, preprocess_function=None, augmentation_function=None, size_factor=10):
        '''
        Initialization
        repetitions -- int/list/tuple, range: (1~10), repetition ids to load data
                    { int: range(1,repetitions+1), list: repetitions, tuple: range(*repetitions) }
        input_dir -- str, dataset directory
        subject -- int, range: (1~27), subject number
        classes -- int/list/tuple, range: (0~52), which classes to load.
                { int: range(0,classes), list: classes, tuple: range(*classes) }
        data_type -- 'raw' or 'rms', type of data to load
        emg_channels -- int, number of channels of emg data
        window_size_H -- int, range: (1~), sliding window's H
        window_size_W -- int/list/tuple, range: (1~emg_data_channels), sliding window's W, which channels
                    emg data to load. { int: range(0,W), list: W, tuple: range(*W) }
        window_step -- int, range: (1~), sliding window's step
        drop_last -- bool, when sliding window, True: drop last emg data; False: no drop last emg data
        normalize -- 'min_max_normalize'or'mean_std_normalize', type of emg data to normalize
        transform -- torch.transform, data preprocess
        preprocess_function -- dict, {func1:{params}, func2:{params}}, data preprocess to apply before augmentation.
                                tips: The type of func is not 'str' but function from data_preprocessing.py
        augmentation_function -- dict, {func1:{params}, func2:{params}}, data augmentation.
                                tips: The type of func1 is not 'str' but function from data_augmentation.py
        size_factor -- int, how many augmentated data are generated for each raw data
        '''
        self.X, self.y, self.r = self.load_emg_data(input_dir=input_dir, subject=subject, classes=classes,
                                                    data_type=data_type, repetitions=repetitions,
                                                    preprocess_function=preprocess_function,
                                                    augmentation_function=augmentation_function,
                                                    size_factor=size_factor)
        self.X_image, self.y_image = self.emg_to_image(self.X, self.y, emg_channels, window_size_H, window_size_W,
                                                       window_step, drop_last, normalize)
        self.transform = transform

    def __getitem__(self, index):
        image, label = self.X_image[index], self.y_image[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.X_image)

    @staticmethod
    def load_emg_data(input_dir, subject, data_type, repetitions, classes, preprocess_function, augmentation_function,
                      size_factor):
        ''' load emg data from designated location'''
        assert os.path.exists(input_dir), 'The input path \'{}\' not exist! Please run \'create_dataset.py\' first.'\
            .format(input_dir)

        assert 1 <= subject <= 27, 'The subject number of Ninapro DB1 requires 1 <= subject <= 27'

        if isinstance(repetitions, int):
            repetitions = [i for i in range(1, repetitions+1)]
        elif isinstance(repetitions, tuple):
            repetitions = [i for i in range(*repetitions)]
        elif isinstance(repetitions, list):
            repetitions = np.unique(repetitions)
        else:
            raise ValueError('The type of input parameter \'repetitions\' must be \'int\', `\'tuple\' or \'tuple\'!')
        assert all([True if 1 <= i <= 10 else False for i in repetitions]), \
            'Require 1 <= i <= 10 for i in repetitions'

        if isinstance(classes, int):
            classes = [i for i in range(classes)]
        elif isinstance(classes, tuple):
            classes = [i for i in range(*classes)]
        elif isinstance(classes, list):
            classes = np.unique(classes)
        else:
            raise ValueError('The type of input parameter \'classes\' must be \'int\', `\'tuple\' or \'tuple\'!')
        assert all([True if 0 <= i <= 52 else False for i in classes]), \
            'Require 0 <= i <= 52 for i in classes'

        X, y, r = [], [], []

        for label in [i for i in classes if i != 0]:
            for rep in repetitions:
                file = '{}/subject-{:02d}/gesture-{:02d}/{}/rep-{:02d}.mat'.format(input_dir, subject, int(label),
                                                                                   data_type, int(rep))
                data = scipy.io.loadmat(file)
                x = data['emg']

                if preprocess_function is not None:
                    for func, params in zip(preprocess_function.keys(), preprocess_function.values()):
                        x = func(x, **params)

                X.append(x)
                y.append(int(np.squeeze(data['stimulus'])[0]))
                r.append(int(np.squeeze(data['repetition'])[0]))

        if 0 in classes:
            classes_no_rest = [i for i in classes if i != 0]
            if len(classes_no_rest) < len(repetitions):
                rest_rep_groups = list(zip(np.random.choice(repetitions, len(repetitions), replace=False),
                                           np.random.choice([i for i in classes if i != 0], len(repetitions),
                                                            replace=True)))
            else:
                rest_rep_groups = list(zip(np.random.choice(repetitions, len(repetitions), replace=False),
                                           np.random.choice([i for i in classes if i != 0], len(repetitions),
                                                            replace=False)))
            for rep, label in rest_rep_groups:
                file = '{}/subject-{:02d}/gesture-00/{}/rep-{:02d}_{:02d}.mat'.format(input_dir, subject, data_type,
                                                                                      int(rep), int(label))
                data = scipy.io.loadmat(file)
                x = data['emg']

                if preprocess_function is not None:
                    for func, params in zip(preprocess_function.keys(), preprocess_function.values()):
                        x = func(x, **params)

                X.append(x)
                y.append(int(np.squeeze(data['stimulus'])[0]))
                r.append(int(np.squeeze(data['repetition'])[0]))

        X_aug, y_agu, r_aug = [], [], []
        for i in range(len(X)):
            # augmented samples
            if augmentation_function is not None:
                for _ in range(size_factor):
                    x = np.copy(X[i])
                    for func, params in zip(augmentation_function.keys(), augmentation_function.values()):
                        x = func(x, **params)
                    X_aug.append(x)
                    y_agu.append(y[i])
                    r_aug.append(r[i])

            # raw samples, both are needed.
            X_aug.append(X[i])
            y_agu.append(y[i])
            r_aug.append(r[i])

        return X_aug, y_agu, r_aug

    @staticmethod
    def emg_to_image(data, y, emg_channels, window_size_H, window_size_W, window_step, drop_last, normalize):
        '''
        split emg data into image by Sliding window
        data -- list, emg data
        y -- list, emg label corresponds to emg data
        '''
        assert len(data) == len(y), 'The length of data and y should be equal!'

        if isinstance(window_size_W, int):
            window_size_W = [i for i in range(window_size_W)]
        elif isinstance(window_size_W, tuple):
            window_size_W = [i for i in range(*window_size_W)]
        elif isinstance(window_size_W, list):
            window_size_W, idx = np.unique(window_size_W, return_index=True)
            window_size_W = window_size_W[np.argsort(idx)]  # keep the original order
        else:
            raise ValueError('The type of input parameter \'window_size_W\' must be \'int\', `\'tuple\' or \'tuple\'!')

        assert all([True if 0 <= i < emg_channels else False for i in window_size_W]), \
            'Require 0 <= i < emg_channels for i in window_size_W'

        assert window_size_H > 0, 'Require window_size_H > 0'
        assert window_step > 0, 'Require window_step > 0'
        if window_size_H >= min([len(data[i]) for i in range(len(data))]):
            raise ValueError('The window_size should be less than the shortest length of the data: {}'
                             .format(min([len(data[i]) for i in range(len(data))])))

        data_offsets = []
        for i in range(len(data)):
            j_list = []
            for j in range(0, len(data[i]) - window_size_H, window_step):
                j_list.append(j)
                data_offsets.append((i, j))
            if not drop_last:
                if (j_list[-1] + window_size_H) < len(data[i]):
                    data_offsets.append((i, len(data[i]) - window_size_H))

        data_image, y_image = [], []
        for i, j in data_offsets:
            x = np.copy(data[i][j:j + window_size_H])
            x = x[:, window_size_W].copy()  # keep the specified column data

            if normalize == 'min_max_normalize':  # Normalization 最小最大归一化
                x = (x - x.min()) / (x.max() - x.min())
            elif normalize == 'mean_std_normalize':  # 标准化/z值归一化
                x = (x - x.mean()) / (x.std())
            elif normalize == 'mean_normalize':  # 均值归一化
                x = (x - x.mean()) / (x.max() - x.min())
            elif normalize == 'max_abs_normalize':  # 最大绝对值归一化
                x = x / (np.abs(x).max())
            elif normalize == 'robust_normalize':  # 稳健标准化
                lower_q = np.quantile(x, 0.25, interpolation='lower')  # 下四分位数
                higher_q = np.quantile(x, 0.75, interpolation='higher')  # 上四分位数
                x = (x - np.median(x)) / (higher_q - lower_q)
            elif normalize is None:
                pass  # No data normalization!
            else:
                supported_method = ['min_max_normalize', 'mean_std_normalize', 'mean_normalize', 'max_abs_normalize',
                                    'robust_normalize']
                raise Exception('The current standardized method is not currently supported! Found \'{}\', expected {}.'
                                .format(normalize, supported_method))

            expected_dim = [window_size_H, len(window_size_W), 1]
            if np.prod(x.shape) == np.prod(expected_dim):
                x = np.reshape(x, expected_dim)  # [H, W, C]-->[C, H ,W] by transform.ToTensor
            else:
                raise Exception('Generated sample dimension mismatch! Found {}, expected {}.'.format(x.shape,
                                                                                                     expected_dim))
            y_image.append(y[i])
            data_image.append(x)

        return data_image, y_image

    @staticmethod
    def min_max_normalize(x):
        return (x - x.min()) / (x.max() - x.min())

    @staticmethod
    def mean_std_normalize(x):
        return (x - x.mean()) / (x.std())

    @staticmethod
    def mean_normalize(x):
        return (x - x.mean()) / (x.max() - x.min())

    @staticmethod
    def max_abs_normalize(x):
        return x / (np.abs(x).max())

    @staticmethod
    def robust_normalize(x):
        lower_q = np.quantile(x, 0.25, interpolation='lower')  # 下四分位数
        higher_q = np.quantile(x, 0.75, interpolation='higher')  # 上四分位数
        return (x - np.median(x)) / (higher_q - lower_q)


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from torchvision import transforms

    preprocess = {low_pass_filter: {'f': 1, 'fs': 100}}
    my_augmentation_function = {time_warp: {'sigma': 0.2}, mag_warp: {'sigma': 0.2}, jitter: {'snr_db': 25}}
    data_set = NinaproDataset(repetitions=[1, 3, 4, 6, 8, 9, 10], classes=53, subject=1,
                              input_dir='./dataset/Ninapro-DB1-Proc', data_type='raw',
                              emg_channels=10, window_size_H=15, window_size_W=10, window_step=15, drop_last=True,
                              normalize=None,
                              transform=transforms.ToTensor(), preprocess_function=preprocess,
                              augmentation_function=my_augmentation_function,
                              size_factor=10)
    data_loader = DataLoader(data_set, batch_size=300, shuffle=True, drop_last=True, num_workers=0)

    print(len(data_set))
    for batch_idx, (data, target) in enumerate(data_loader):
        print(batch_idx, data.shape, target.shape)

