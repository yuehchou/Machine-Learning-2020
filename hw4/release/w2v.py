import os
import numpy as np
import pandas as pd
import argparse
from gensim.models import word2vec

path_prefix = './../data/'

train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')
testing_data = os.path.join(path_prefix, 'testing_data.txt')

w2v_path = os.path.join(path_prefix, 'w2v_all.model')

print("loading training data ...")
train_x, y = load_training_data(train_with_label)
train_x_no_label = load_training_data(train_no_label)

print("loading testing data ...")
test_x = load_testing_data(testing_data)


model = word2vec.Word2Vec(
    train_x + train_x_no_label + test_x,
    size=50,
    window=5,
    min_count=5,
    workers=12,
    iter=30,
    sg=1
)

print("saving model ...")
model.save(os.path.join(path_prefix, 'w2v_all.model'))
