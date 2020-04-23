import os, cv2, time, torch
import numpy as np
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import *
from training_model import *

workspace_dir = None
lr = 0.00005
batch_size = 128
num_epoch = 1
patience = None
save_model_path = None
results_fpath = None

early_stopping = EarlyStopping(patience=patience, verbose=True)
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

print("Reading data")
train_x, train_y = readfile_resnet(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))
val_x, val_y = readfile_resnet(os.path.join(workspace_dir, "validation"), True)
print("Size of validation data = {}".format(len(val_x)))

total_train_x = np.concatenate((train_x, val_x))
total_train_y = np.concatenate((train_y, val_y))
print("Size of total training data = {}".format(len(total_train_x)))

cv_train_x, cv_train_y, cv_val_x, cv_val_y = cv_split(
    total_train_x,
    total_train_y,
    10,
    10
)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.RandomResizedCrop(128),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    normalize
])

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.CenterCrop(128),
    transforms.Resize(224),
    transforms.ToTensor(),
    normalize
])

train_set = ImgDataset(cv_train_x, cv_train_y, train_transform)
val_set = ImgDataset(cv_val_x, cv_val_y, val_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)


model = models.resnet152(pretrained=True)
model.cuda()
model = nn.DataParallel(model)
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr) # optimizer 使用 Adam

for epoch in range(num_epoch):
    epoch_start_time = time.time()
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    model.train() # 確保 model 是在 train model (開啟 Dropout 等...)
    for i, data in enumerate(train_loader):
        optimizer.zero_grad() # 用 optimizer 將 model 參數的 gradient 歸零
        train_pred = model(data[0].cuda()) # 利用 model 得到預測的機率分佈 這邊實際上就是去呼叫 model 的 forward 函數
        batch_loss = loss(train_pred, data[1].cuda()) # 計算 loss （注意 prediction 跟 label 必須同時在 CPU 或是 GPU 上）
        batch_loss.backward() # 利用 back propagation 算出每個參數的 gradient
        optimizer.step() # 以 optimizer 用 gradient 更新參數值

        train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
        train_loss += batch_loss.item()

    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            val_pred = model(data[0].cuda())
            batch_loss = loss(val_pred, data[1].cuda())

            val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            val_loss += batch_loss.item()

        #將結果 print 出來
        print('[%03d/%03d] %2.2f sec(s) Train Acc: %3.6f Loss: %3.6f | Val Acc: %3.6f loss: %3.6f' % \
            (epoch + 1, num_epoch, time.time()-epoch_start_time, \
             train_acc/train_set.__len__(), train_loss/train_set.__len__(), val_acc/val_set.__len__(), val_loss/val_set.__len__()))

    early_stopping(val_acc/val_set.__len__(), model, save_model_path)

    if early_stopping.early_stop:
        print("Early stopping")
        break

print('Load model from {}'.format(save_model_path))
# load the last checkpoint with the best model
model.load_state_dict(torch.load(save_model_path))

model.eval()
prediction = []
answer = []
with torch.no_grad():
    for i, data in enumerate(val_loader):
        val_pred = model(data[0].cuda())
        val_label = np.argmax(val_pred.cpu().data.numpy(), axis=1)
        for y in val_label:
            prediction.append(y)
        for y in data[1]:
            answer.append(y)

with open(results_fpath, 'w') as f:
    f.write('Id,Category,Answer\n')
    for i, y in  enumerate(prediction):
        f.write('{},{},{}\n'.format(i, y, answer[i]))
