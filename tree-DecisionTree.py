import numpy as np
from sklearn.utils import shuffle
from data_loader.data_loaders import NinaPro_Dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

train = NinaPro_Dataset(DB=1, subject=3, trainRepeatIndex=[0,1,3,4,6,8,9,10])
tests = NinaPro_Dataset(DB=1, subject=3, trainRepeatIndex=[2,5,7])
print('Dimension of feature space: ', train.Xs.shape)
print('Tests samples numbers: ', tests.Xs.shape)

clf = DecisionTreeClassifier(max_depth=13, min_samples_split=3)
clf.fit(train.Xs, train.Ys)
print('Train: ', clf.score(train.Xs, train.Ys))
print('Tests: ', clf.score(tests.Xs, tests.Ys))