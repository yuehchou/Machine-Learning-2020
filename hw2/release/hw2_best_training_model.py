import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model

np.random.seed(0)
X_train_fpath = './X_train'
Y_train_fpath = './Y_train'
X_test_fpath = './X_test'
output_fpath = './output_{}.csv'

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

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _ = _normalize(X_test,
	train = False,
	specified_column = None,
	X_mean = X_mean,
	X_std = X_std
)
    
# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(
	X_train,
	Y_train,
	dev_ratio = dev_ratio
)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]

print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

X_train_best = pd.DataFrame(X_train)
Y_train_best = pd.DataFrame(Y_train)
X_dev_best = pd.DataFrame(X_dev)
Y_dev_best = pd.DataFrame(Y_dev)
X_test_best = pd.DataFrame(X_test)

importance = list(np.load('./importance.npy'))

X_train_selected = X_train_best.loc[:,importance]
X_dev_selected = X_dev_best.loc[:,importance]
X_test_selected = X_test_best.loc[:,importance]

input_dim = X_train_selected.shape[1]

# define model
model = Sequential()
model.add(Dense(24, input_dim=input_dim, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(
	loss = 'binary_crossentropy',
	optimizer = 'adam',
	metrics = ['accuracy']
)

# simple early stopping
early_stopping = EarlyStopping(
	monitor = 'val_loss',
	patience = 10,
	verbose = 1
)

# fit model
history = model.fit(
	X_train_selected,
	Y_train_best,
	validation_data = (X_dev_selected, Y_dev_best),
	batch_size = 200,
	epochs = 4000,
	verbose = 1,
	callbacks = [early_stopping]
)

# evaluate the model
_, train_acc = model.evaluate(X_train_selected, Y_train_best, verbose=0)
_, test_acc = model.evaluate(X_dev_selected, Y_dev_best, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

Y_test_pred = model.predict(X_test_selected, verbose=1)
Y_test_pred=np.asarray(Y_test_pred).reshape(-1)
Y_test_pred = np.round(Y_test_pred).astype(np.int)

# Predict testing labels
predictions = Y_test_pred
with open(output_fpath.format('prediction'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))