import torch, time
import torch.optim as optim
import numpy as np
import torch.nn as nn
from bert_utils import TextDataset
from pytorchtools import EarlyStopping
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from bert_training import *
from bert_model import *


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# max_length_sentence = tokenizer.max_len_single_sentence
max_length_sentence = 50

bert = BertModel.from_pretrained('bert-base-uncased')

np_text_fpath = './../data/np_training_x.npy'
np_label_fpath = './../data/np_training_y.npy'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 5000 # checkpoint_val_acc_0_8318.pt
N_EPOCHS = 70
learning_rate = 0.001
gamma = 0.5
patience = 30
milestones = [10, 20, 30, 40, 50]
early_stopping = EarlyStopping(patience=patience, verbose=True)

criterion = nn.BCEWithLogitsLoss()

HIDDEN_DIM = 256
# HIDDEN_DIM = 512
OUTPUT_DIM = 1
# N_LAYERS = 2
N_LAYERS = 4
BIDIRECTIONAL = True
# DROPOUT = 0.25
DROPOUT = 0.2

model = BERTGRUSentiment(bert,
                         HIDDEN_DIM,
                         OUTPUT_DIM,
                         N_LAYERS,
                         BIDIRECTIONAL,
                         DROPOUT)
model = model.to(device)
model = nn.DataParallel(model)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = criterion.to(device)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

cut_num = 180000

save_checkpoint_path = './../model/checkpoint.pt'

np_text = np.load(np_text_fpath)
np_label = np.load(np_label_fpath)

X_train, X_val, y_train, y_val = np_text[:cut_num], np_text[cut_num:], np_label[:cut_num], np_label[cut_num:]

train_set = TextDataset(X_train, y_train)
val_set = TextDataset(X_val, y_val)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# best_val_loss = float('inf')
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

    # early_stopping(val_loss, model, save_checkpoint_path)
    early_stopping(val_acc, model, save_checkpoint_path)

    if early_stopping.early_stop:
        print("Early stopping")
        print("Best validation accuracy: {:.2f}%".format(best_val_acc*100))
        break
