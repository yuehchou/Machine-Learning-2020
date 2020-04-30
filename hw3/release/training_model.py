import numpy as np
import torch, time
import torch.nn as nn
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import *

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.dnn = nn.Sequential(
            nn.Linear(128, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(512, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(2048, 4096),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(4096, 6120),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(6120, 3072),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(3072, 2048),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(2048, 1024),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(48 * 8 * 8, 24 * 4 * 4),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(24 * 4 * 4, 12 * 2 * 2),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(12 * 2 * 2, 11)
        )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),  # [16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, 1, 1),  # [16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [16, 64, 64]

            nn.Conv2d(16, 32, 3, 1, 1), # [32 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), # [32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [32, 32, 32]

            nn.Conv2d(32, 64, 3, 1, 1), # [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), # [64, 32, 32]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 16, 16]
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*16*16, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 11),
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.Linear(1000, 11),
        )

    def forward(self, x):
        # return self.dnn(x)

        # out = self.cnn2(x)
        # out = out.view(out.size()[0], -1)
        # return self.fc2(out)

        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

def train_model(
        model,
        batch_size,
        num_epoch,
        loss,
        optimizer,
        train_set,
        val_set,
        save_model_path,
        patience=10
):
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    temp_patience = 0
    temp_val_acc = 0
    temp_val_best_acc = 0
    temp_val_loss = np.inf
    temp_model = None
    for epoch in range(num_epoch):
        epoch_start_time = time.time()
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
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

        # early_stopping(val_loss/val_set.__len__(), model, save_model_path)
        early_stopping(val_acc/val_set.__len__(), model, save_model_path)

        if temp_val_acc < val_acc/val_set.__len__():
            temp_val_acc = val_acc/val_set.__len__()
            temp_val_best_acc = val_acc/val_set.__len__()

        if early_stopping.early_stop:
            print("Early stopping")
            break
    return temp_val_best_acc


def predict(model, batch_size, test_x, results_fpath):

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(150),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456 , 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    model.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.cuda())
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    with open(results_fpath, 'w') as f:
        f.write('Id,Category\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, y))
