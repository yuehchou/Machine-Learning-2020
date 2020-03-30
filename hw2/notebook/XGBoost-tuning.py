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
learning_rate = 0.02
tuning_n_estimators = 150
n_estimators = 200
objective = 'binary:logistic'
cv_score = 'roc_auc'
cv_n_fold = 10
cv_metrics = 'auc'
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
    early_stopping_rounds = 200
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

print('Start tuning...')

print("[XGBoost] tuning max_depth and min_child_weight...")
param_test = {
    'max_depth' : range(1,6,1),
    'min_child_weight' : range(1,6,1)
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        subsample = 0.5,
        colsample_bytree = 0.5,
        objective = objective,
        nthread = 4,
        scale_pos_weight = 1,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
max_depth = grid_search.best_params_['max_depth']
min_child_weight = grid_search.best_params_['min_child_weight']

# #################################################
# Tuning gamma
print("[XGBoost] tuning gamma...")
param_test = {
    'gamma' : [i/10.0 for i in range(11)]
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        subsample = 0.5,
        colsample_bytree = 0.5,
        objective = objective,
        nthread = 5,
        scale_pos_weight = 1,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
gamma = grid_search.best_params_['gamma']

# #################################################
# Tuning subsample and colsample_bytree
print("[XGBoost] tuning subsample and colsample_bytree...")
param_test = {
    'subsample' : [i/10.0 for i in range(1,11)],
    'colsample_bytree' : [i/10.0 for i in range(1,11)]
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        objective = objective,
        nthread = 5,
        scale_pos_weight = 1,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
subsample = grid_search.best_params_['subsample']
colsample_bytree = grid_search.best_params_['colsample_bytree']

# #################################################
# Tuning reg_alpha
print("[XGBoost] tuning reg_alpha...")
param_test = {
    'reg_alpha': range(11)
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        objective = objective,
        nthread = 5,
        scale_pos_weight = 1,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
reg_alpha = grid_search.best_params_['reg_alpha']

# #################################################
# Tuning nthread
param_test = {
    'nthread': range(1,11)
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        reg_alpha = reg_alpha,
        objective = objective,
        scale_pos_weight = 1,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
nthread = grid_search.best_params_['nthread']

# #################################################
# Tuning scale_pos_weight
print("[XGBoost] tuning scale_pos_weight...")
param_test = {
    'scale_pos_weight': np.arange(1, 2.5, 0.5)
}
grid_search = GridSearchCV(estimator = XGBClassifier(
        learning_rate = learning_rate,
        n_estimators = tuning_n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        gamma = gamma,
        subsample = subsample,
        colsample_bytree = colsample_bytree,
        reg_alpha = reg_alpha,
        objective = objective,
        nthread = nthread,
        seed = 27
    ),
    param_grid = param_test,
    scoring = 'roc_auc',
    n_jobs = n_jobs,
    iid = False,
    cv = cv_n_fold,
#     early_stopping_rounds = 200
)
grid_search.fit(X_train, Y_train)
scale_pos_weight = grid_search.best_params_['scale_pos_weight']

# #################################################
# Training
print('start training...')
xgb_params = XGBClassifier(
    learning_rate = learning_rate,
    n_estimators = 1000,
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    gamma = gamma,
    subsample = subsample,
    reg_alpha = reg_alpha,
    colsample_bytree = colsample_bytree,
    objective = objective,
    nthread = nthread,
    scale_pos_weight = scale_pos_weight,
    seed = 27
)
xgb1 = XGBClassifier(xgb_params.get_xgb_params())
cvresult = modelfit(xgb1, X_train, Y_train)

xgb_params = XGBClassifier(
    learning_rate = learning_rate,
    n_estimators = len(cvresult),
    max_depth = max_depth,
    min_child_weight = min_child_weight,
    gamma = gamma,
    subsample = subsample,
    reg_alpha = reg_alpha,
    colsample_bytree = colsample_bytree,
    objective = objective,
    nthread = nthread,
    scale_pos_weight = scale_pos_weight,
    seed = 27
)
xgb1 = XGBClassifier(xgb_params.get_xgb_params())
xgb1.fit(X_train, Y_train)
    
# #################################################
# Predict testing labels
predictions = xgb1.predict(X_test)
with open(output_fpath.format('XGB'), 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, int(label)))