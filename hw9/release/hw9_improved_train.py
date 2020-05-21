import sys
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch import optim
from torch.utils.data import DataLoader
from utils import preprocess
from utils import Image_Dataset
from utils import same_seeds
from model import AE_improved

n_epoch = 130
same_seeds(0)

trainX_path = sys.argv[1]
checkpoints_path = sys.argv[2]

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])

trainX = np.load(trainX_path)
trainX_preprocessed = preprocess(trainX)
img_dataset = Image_Dataset(trainX, train_transform)
img_dataloader = DataLoader(img_dataset, batch_size=64, shuffle=True)

model = AE_improved().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

model.train()

epoch_loss = 0
for epoch in range(n_epoch):
    epoch_loss = 0
    for data in img_dataloader:
        img = data
        img = img.cuda()

        output1, output = model(img)
        loss = criterion(output, img)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            torch.save(model.state_dict(), '{}_{}.pth'.format(checkpoints_path[:-4], epoch+1))

        epoch_loss += loss.item()

    print('epoch [{}/{}], loss:{:.5f}'.format(epoch+1, n_epoch, epoch_loss))

torch.save(model.state_dict(), checkpoints_path)
