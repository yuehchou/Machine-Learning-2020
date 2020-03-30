import sys, keras
import numpy as np
import pandas as pd
from keras.models import load_model

X_train_fpath = sys.argv[3]
Y_train_fpath = sys.argv[4]
X_test_fpath = sys.argv[5]
output_fpath = sys.argv[6]

def _normalize(
    X,
    train = True,
    specified_column = None,
    X_mean = None,
    X_std = None
):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

X_mean = np.load('./X_mean.npy')
X_std = np.load('./X_std.npy')
importance = list(np.load('./importance.npy'))
X_test, _, _ = _normalize(
    X_test,
    train = False,
    specified_column = None,
    X_mean = X_mean,
    X_std = X_std
)
X_test = pd.DataFrame(X_test)
X_test_selected = X_test.loc[:, importance]

model = load_model('./model.h5')
Y_test_pred = model.predict(X_test_selected, verbose=1)
Y_test_pred = np.array(Y_test_pred).reshape(-1)
Y_test_pred = np.round(Y_test_pred).astype(np.int)

# Predict testing labels
predictions = Y_test_pred
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))
