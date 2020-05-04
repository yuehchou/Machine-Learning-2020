import torch, time
import pandas as pd
import torch.optim as optim
import numpy as np
import torch.nn as nn
from bert_utils import TextDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from bert_training import *
from bert_model import *

def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_length_sentence = tokenizer.max_len_single_sentence
max_length_sentence = 50

bert = BertModel.from_pretrained('bert-base-uncased')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

load_checkpoint_path = './../model/checkpoint_val_acc_0_8357.pt'
np_text_fpath = './../data/np_testing.npy'
results_fpath = './../results/prediction_val_acc_0_8357.csv'

batch_size = 5000 # checkpoint_val_acc_0_8318.pt

criterion = nn.BCEWithLogitsLoss()

HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.25

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
model = model.to(device)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_checkpoint_path))

np_text = np.load(np_text_fpath)
test_set = TextDataset(np_text)
test_loader = DataLoader(test_set, batch_size=batch_size)

model.eval()

print('Predicting...')
ret_output = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        inputs = inputs.to(device, dtype=torch.long)
        outputs = model(inputs)
        outputs = outputs.squeeze()
        ret_output += outputs.tolist()
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
