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
w2v_path = './../data/w2v_all.model'
save_checkpoint_path = './../model/checkpoint_bs1024_lstm.pt'
save_history_path = './../results/history_bs1024_lstm.csv'

print("loading data ...")
train_x, y = load_training_data(train_with_label)
preprocess = Preprocess(train_x, max_length_sentence, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

cut_num = 190000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch_size = 5000 # checkpoint_val_acc_0_8318.pt
batch_size = 2048
N_EPOCHS = 100
learning_rate = 0.001
gamma = 0.1
patience = 20
milestones = [60]
early_stopping = EarlyStopping(patience=patience, verbose=True)

history = {}
history['train_acc'] = []
history['val_acc'] = []
history['train_loss'] = []
history['val_loss'] = []

criterion = nn.BCEWithLogitsLoss()

embedding_dim = 50
HIDDEN_DIM = 512
N_LAYERS = 6
DROPOUT = 0.2

model = LSTM(
    embedding,
    embedding_dim=embedding_dim,
    hidden_dim=HIDDEN_DIM,
    num_layers=N_LAYERS,
    dropout=DROPOUT,
    fix_embedding=True
)

model = model.to(device)
model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00002)
criterion = criterion.to(device)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

X_train, X_val, y_train, y_val = train_x[:cut_num], train_x[cut_num:], y[:cut_num], y[cut_num:]

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

    history['train_acc'].append(train_acc)
    history['val_acc'].append(val_acc)
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)

df = pd.DataFrame(history)
df.to_csv(save_history_path)
print("Save history to {}".format(save_history_path))
