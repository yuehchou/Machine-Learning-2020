import os, cv2, time, torch, sys
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from torch.optim import lr_scheduler
from data_preprocessing import *
from training_model import *

torch.cuda.empty_cache()

save_model_path = None
workspace_dir = sys.argv[1]

lr = 0.0005
gamma = 0.5
batch_size = 256
num_epoch = 300
patience = 30
milestones = [200]

early_stopping = EarlyStopping(patience=patience, verbose=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

total_train_x = np.concatenate((train_x, val_x))
total_train_y = np.concatenate((train_y, val_y))
print("Size of total training data = {}".format(len(total_train_x)))

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize,
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    normalize
])

train_set = ImgDataset(train_x, train_y, train_transform)
val_set = ImgDataset(val_x, val_y, val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

model = Classifier().cuda()
model = nn.DataParallel(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train()
    optimizer.step()
    scheduler.step()
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        train_pred = model(data[0].cuda())
        batch_loss = loss(train_pred, data[1].cuda())
        batch_loss.backward()
        optimizer.step()

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    early_stopping(val_acc/val_set.__len__(), model, save_model_path)

    if early_stopping.early_stop:
        print("Early stopping")
        break
