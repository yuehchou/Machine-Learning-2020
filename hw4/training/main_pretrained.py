import torch, time
import torch.optim as optim
import numpy as np
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
load_checkpoint_path = './../model/checkpoint_bs1024.pt'
save_checkpoint_path = './../model/checkpoint_bs1024_pretrained.pt'

print("loading data ...")
train_x, y = load_training_data(train_with_label)
preprocess = Preprocess(train_x, max_length_sentence, w2v_path=w2v_path)
embedding = preprocess.make_embedding(load=True)
train_x = preprocess.sentence_word2idx()
y = preprocess.labels_to_tensor(y)

cut_num = 180000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch_size = 5000 # checkpoint_val_acc_0_8318.pt
batch_size = 1024 # checkpoint_val_acc_0_8318.pt
N_EPOCHS = 30
learning_rate = 0.00001
gamma = 0.5
patience = 10
milestones = [5, 10, 15, 20]
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
