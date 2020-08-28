import numpy as np
from sklearn.utils import shuffle
from data_loader.data_loaders import NinaPro_Dataset
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

nina = NinaPro_Dataset(DB=1, subject=2, trainRepeatIndex=[0,1,2,3,4])
print('Dimension of feature space: ', nina.Xs.shape)

# necessary steps for NaN, too large value, and so on. 
Xtrain, Xtests, ytrain, ytests = train_test_split(nina.Xs, nina.Ys.squeeze(1), shuffle=False)

clf = RandomForestClassifier(oob_score=True)
clf.fit(Xtrain, ytrain)

print('Train: ', clf.score(Xtrain, ytrain))
print('Tests: ', clf.score(Xtests, ytests))