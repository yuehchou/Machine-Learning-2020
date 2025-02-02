import os, sys
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

csv_fpath = './validation_prediction_and_answer.csv'
save_fig_path = os.path.join(sys.argv[2], 'confusion_matrix.png')

def plot_confusion_matrix(
   cm,
   classes,
   save_fpath,
   normalize=False,
   title='Confusion matrix',
   cmap=plt.cm.Blues
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(save_fpath)


df = pd.read_csv(csv_fpath)
pred = np.array(df['Category'])
ans  = np.array(df['Answer'])

cnf_matrix = confusion_matrix(ans, pred)

plot_confusion_matrix(
   cnf_matrix[:11,:11],
   classes=range(11),
   save_fpath=save_fig_path,
   normalize=True,
   title='Confusion matrix',
   cmap=plt.cm.viridis
)
