# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
import numpy as np
from data_loader.data_loaders import NinaPro_Dataset
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# %%
train = NinaPro_Dataset(DB=1, subject=15, trainRepeatIndex=[0,1,3,4,6,8,9,10])
tests = NinaPro_Dataset(DB=1, subject=15, trainRepeatIndex=[2,5,7])
print('Dimension of feature space: ', train.Xs.shape)
print('Tests samples numbers: ', tests.Xs.shape)


# %%
# clf = DecisionTreeClassifier()
clf = RandomForestClassifier(max_depth=100, n_estimators=3000)
clf.fit(train.Xs, train.Ys)
print('Train: ', clf.score(train.Xs, train.Ys))
print('Tests: ', clf.score(tests.Xs, tests.Ys))
