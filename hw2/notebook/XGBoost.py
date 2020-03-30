import numpy as np
from sklearn.metrics import accuracy_score
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import roc_auc_score, roc_curve, auc, accuracy_score
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn import metrics
import matplotlib.pyplot as pyplot
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 8

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

# ####################################################
# grid search and cross validation setting
learning_rate = 0.001
tuning_n_estimators = 5000
n_estimators = 100000
objective = 'binary:logistic'
cv_score = 'accuracy'
cv_n_fold = 10
cv_metrics = 'accuracy'
num_boost_round = 5000
early_stopping_rounds = 1000
show_stdv = True
n_jobs = -1

# ####################################################
def modelfit(
    alg,
    Train_X,
    Train_Y,
    useTrainCV = True,
    cv_folds = 10,
    early_stopping_rounds = 20
):

    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(
            Train_X,
            label = Train_Y
        )
        cvresult = xgb.cv(
            xgb_param,
            xgtrain,
            num_boost_round = alg.get_params()['n_estimators'],
            nfold = cv_folds,
            metrics = 'auc',
            early_stopping_rounds = early_stopping_rounds
        )
        alg.set_params(n_estimators=cvresult.shape[0])

    #Fit the algorithm on the data
    alg.fit(Train_X, Train_Y, eval_metric = 'auc')

    #Predict training set:
    dtrain_predictions = alg.predict(Train_X)
    dtrain_predprob = alg.predict_proba(Train_X)[:,1]

    #Print model report:
    print("\nModel Report")
    print("Accuracy : %.4g" % metrics.accuracy_score(Train_Y, dtrain_predictions))
    print("AUC Score (Train): %f" % metrics.roc_auc_score(Train_Y, dtrain_predprob))
    print(alg)
    print("the best:")
    print(cvresult.shape[0])
    plot_importance(alg)
    plt.show()
 
    if useTrainCV:
        return cvresult



def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
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

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

# Normalize training and testing data
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

# Split data into training set and development set
dev_ratio = 0.
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))

# #################################################
# Training
print('start training...')
xgb1 = XGBClassifier(
    n_estimators = 1000,
    learning_rate = 0.02,
    max_depth = 5,
    min_child_weight = 1,
    gamma = 0.9,
    subsample = 1.,
    reg_alpha = 0,
    colsample_bytree = 0.3,
    objective = 'binary:logistic',
    nthread = 1,
    scale_pos_weight = 1.5,
    seed = 27,
    n_jobs = -1
)
cvresult = modelfit(xgb1, X_train, Y_train)
xgb1 = XGBClassifier(
    n_estimators = len(cvresult),
    learning_rate = 0.02,
    max_depth = 5,
    min_child_weight = 1,
    gamma = 0.9,
    subsample = 1.,
    reg_alpha = 0,
    colsample_bytree = 0.3,
    objective = 'binary:logistic',
    nthread = 1,
    scale_pos_weight = 1.5,
    seed = 27,
    n_jobs = -1
)
xgb1.fit(X_train, Y_train)

# #################################################
# Predict testing labels
predictions = xgb1.predict(X_test)
with open(output_fpath.format('XGB_tuned'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, int(label)))
