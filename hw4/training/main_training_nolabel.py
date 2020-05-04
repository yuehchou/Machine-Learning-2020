import torch, time
import torch.optim as optim
import numpy as np
import pandas as pd
import torch.nn as nn
from utils import *
from preprocess import Preprocess
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from model import *
from bert_training import *
from data import TwitterDataset

max_length_sentence = 50
fix_embedding = True

train_with_label = './../data/training_label.txt'
train_nolabel = './../data/training_nolabel.txt'
train_nolabel_label = './../results/prediction_train_nolabel.csv'
w2v_path = './../data/w2v_all.model'
load_checkpoint_path = './../model/checkpoint_bs2048.pt'
save_checkpoint_path = './../model/checkpoint_semi.pt'
history_path = './../results/semi_history.csv'

history = {}
history['train acc'] = []
history['val acc'] = []
history['train loss'] = []
history['val loss'] = []

print("loading data ...")
train_x, y = load_training_data(train_with_label)
preprocess = Preprocess(train_x, max_length_sentence, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

train_nolabel_x = load_training_data(train_nolabel)
preprocess = Preprocess(train_nolabel_x, max_length_sentence, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_nolabel_x = preprocess.sentence_word2idx()
df_train_nolabel_y = pd.read_csv(train_nolabel_label)
train_nolabel_y = list(df_train_nolabel_y['label'])
train_nolabel_y = preprocess.labels_to_tensor(train_nolabel_y)

cut_num = 180000
train_label_num = 20000
# train_nolabel_num = 50000
train_nolabel_num = 1000000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 2048
N_EPOCHS = 100
learning_rate = 0.01
gamma = 0.1
patience = 50
milestones = [10, 80]
early_stopping = EarlyStopping(patience=patience, verbose=True)

criterion = nn.BCEWithLogitsLoss()

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00001)
criterion = criterion.to(device)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

# X_train, X_val = train_x[cut_num-train_label_num:cut_num], train_x[cut_num:]
X_train, X_val = train_x[:cut_num], train_x[cut_num:]
# y_train, y_val = y[cut_num-train_label_num:cut_num], y[cut_num:]
y_train, y_val = y[:cut_num], y[cut_num:]

y_train = y_train.tolist()
X_train = X_train.tolist()
train_nolabel_y = train_nolabel_y.tolist()
train_nolabel_x = train_nolabel_x.tolist()
for i in range(len(train_nolabel_y)):
    if i == train_nolabel_num:
        break
    X_train.append(train_nolabel_x[i])
    y_train.append(train_nolabel_y[i])

print("X_train length: {}".format(len(X_train)))
print("y_train length: {}".format(len(y_train)))

# y_train = [int(label) for label in y_train]
y_train = torch.FloatTensor(y_train)
X_train = torch.LongTensor(X_train)

train_set = TwitterDataset(X=X_train, y=y_train)
val_set = TwitterDataset(X=X_val, y=y_val)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

best_val_acc = 0.0

for epoch in range(N_EPOCHS):

    start_time = time.time()

    optimizer.step()
    scheduler.step()
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)

    end_time = time.time()

    if best_val_acc < val_acc:
        best_val_acc = val_acc

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print('Epoch: {} | Epoch Time: {}m {}s'.format(epoch+1, epoch_mins, epoch_secs))
    print('\tTrain Loss: {:.3f} | Train Acc: {:.2f}%'.format(train_loss, train_acc*100))
    print('\t Val. Loss: {:.3f} |  Val. Acc: {:.2f}%'.format(val_loss, val_acc*100))

    early_stopping(val_acc, model, save_checkpoint_path)

    if early_stopping.early_stop:
        print("Early stopping")
        print("Best validation accuracy: {:.2f}%".format(best_val_acc*100))
        break

    history['train acc'].append(train_acc)
    history['val acc'].append(val_acc)
    history['train loss'].append(train_loss)
    history['val loss'].append(val_loss)

df = pd.DataFrame(history)
df.to_csv(history_path, index=False)
print("Save history to {}".format(history_path))
