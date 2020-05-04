import torch, time
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
from preprocess import Preprocess
from torch.utils.data import DataLoader
from utils import *
from data import TwitterDataset
from bert_training import *
from model import *

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

max_length_sentence = 50

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_checkpoint_path = './../model/checkpoint_bs1024.pt'
w2v_path = './../data/w2v_all.model'
testing_data = './../data/testing_data.txt'
results_fpath = './../results/prediction_bs1024.csv'

print("loading data ...")
test_x = load_testing_data(testing_data)
print(len(test_x))
preprocess = Preprocess(test_x, max_length_sentence, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
test_x = preprocess.sentence_word2idx()

# batch_size = 5000 # checkpoint_val_acc_0_8318.pt
# batch_size = 2048
batch_size = 1024

embedding_dim = 50
# HIDDEN_DIM = 256
HIDDEN_DIM = 512
# N_LAYERS = 4
N_LAYERS = 6
# DROPOUT = 0.25
DROPOUT = 0.2

model = GRUSentiment(embedding,
                     embedding_dim,
                     HIDDEN_DIM,
                     N_LAYERS,
                     DROPOUT)

model = model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_checkpoint_path))

test_set = TwitterDataset(X=test_x, y=None)
test_loader = DataLoader(test_set, batch_size=batch_size)

model.eval()
print('Predicting...')
ret_output = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device, dtype=torch.long)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        outputs = outputs.tolist()
        for y in outputs:
            ret_output.append(y)
print('Done')

for i in range(len(ret_output)):
    y = sigmoid(ret_output[i])
    if y >= 0.5:
        ret_output[i] = 1
    else:
        ret_output[i] = 0

tmp = pd.DataFrame({"id":[str(i) for i in range(len(ret_output))],"label":ret_output})
print("save csv ...")
tmp.to_csv(results_fpath, index=False)
print('Save results to {}'.format(results_fpath))
