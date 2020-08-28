import numpy as np
import pandas as pd
from tsfresh import extract_features as tsfresh_Feature_Extract
def MAV(X, param):
    pass

def RMS(X, param):
    pass


def get_tsfresh_features(X=None, \
                         LW=10, \
                         LI=5, \
                         features_Parameters={'abs_energy':None, \
                                              'absolute_sum_of_changes':None, \
                                              'has_duplicate':None, \
                                              'count_below_mean':None}):
    assert LI>0, 'Wrong::LI should larger than 0!!!'
    X = np.asarray(X)
    X_Rows = X.shape[0]
    assert X_Rows > LW, 'Wrong:: LW is larger than X_rows!!!'
    num_Samples = int((X_Rows-LW)/LI) + 1
    X2tsfresh_index = []
    X2tsfresh_id = []
    for i in range(num_Samples):
        X2tsfresh_index.append([j for j in range(i*LI, i*LI+LW)])
        X2tsfresh_id.append([i]*LW)

    X2tsfresh_index = np.asarray(X2tsfresh_index).reshape(num_Samples*LW)
    X2tsfresh_id = np.asarray(X2tsfresh_id).reshape(num_Samples*LW)
    X2tsfresh = pd.DataFrame(X[X2tsfresh_index])
    X2tsfresh['id'] = X2tsfresh_id
    # Xfeature = tsfresh_Feature_Extract(X2tsfresh, column_id='id', default_fc_parameters=features_Parameters)
    Xfeature = tsfresh_Feature_Extract(X2tsfresh, column_id='id', default_fc_parameters=None)

    return np.asarray(Xfeature)
