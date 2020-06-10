

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from utils import same_seeds

import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

source_dataset = ImageFolder('real_or_drawing/train_data', transform=source_transform)
target_dataset = ImageFolder('real_or_drawing/test_data', transform=target_transform)

source_dataloader = DataLoader(source_dataset, batch_size=32*16, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32*16, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128*4, shuffle=False)



import cv2
import matplotlib.pyplot as plt

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
domain_classifier = DomainClassifier().cuda()

class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

optimizer_F = optim.Adam(feature_extractor.parameters())
optimizer_C = optim.Adam(label_predictor.parameters())
optimizer_D = optim.Adam(domain_classifier.parameters())

def train_epoch(source_dataloader, target_dataloader, lamb):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0

    for i, ((source_data, source_label), (target_data, _)) in enumerate(zip(source_dataloader, target_dataloader)):

        source_data = source_data.cuda()
        source_label = source_label.cuda()
        target_data = target_data.cuda()
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        mixed_data = torch.cat([source_data, target_data], dim=0)
        domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(mixed_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        domain_logits = domain_classifier(feature.detach())
        loss = domain_criterion(domain_logits, domain_label)
        running_D_loss+= loss.item()
        loss.backward()
        optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) - lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_D_loss / (i+1), running_F_loss / (i+1), total_hit / total_num

# 訓練200 epochs
for epoch in range(100):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.2)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

for epoch in range(100):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.4)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.6)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=0.8)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=1)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

for epoch in range(200):
    train_D_loss, train_F_loss, train_acc = train_epoch(source_dataloader, target_dataloader, lamb=1.2)

    torch.save(feature_extractor.state_dict(), f'extractor_ada_model.bin')
    torch.save(label_predictor.state_dict(), f'predictor_ada_model.bin')

    print('epoch {:>3d}: train D loss: {:6.4f}, train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_D_loss, train_F_loss, train_acc))

# Import需要的套件
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import time

workspace_dir = 'real_or_drawing/test_data'
path=os.path.join(workspace_dir, "0")

workspace_dir = 'real_or_drawing/test_data'
path=os.path.join(workspace_dir, "0")
def readfile(path, label):
    # label 是一個 boolean variable，代表需不需要回傳 y 值
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 28, 28, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :] = cv2.resize(img,(28, 28))
        if label:
          y[i] = int(file.split("_")[0])
    if label:
      return x, y
    else:
      return x

# training 時做 data augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    #transforms.ToPILImage(),
    transforms.Resize(32),
    #transforms.RandomResizedCrop(128),
    transforms.RandomHorizontalFlip(), # 隨機將圖片水平翻轉
    #transforms.RandomRotation(15), # 隨機旋轉圖片
    transforms.RandomRotation(15),
    transforms.ToTensor(), # 將圖片轉成 Tensor，並把數值 normalize 到 [0,1] (data normalization)
    #normalize,
])
# testing 時不需做 data augmentation
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(),
    #transforms.ToPILImage(),   
    transforms.Resize(32),
    transforms.CenterCrop(32),                                
    transforms.ToTensor(),
    #normalize,
])
class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

workspace_dir = 'real_or_drawing/test_data'
path=os.path.join(workspace_dir, "0")

test_x = readfile(os.path.join(workspace_dir, "0"), False)

import pandas as pd
df = pd.read_csv('pseudo_label.csv', delimiter=',')
label=df["label"].values
len(label)
label=list(label)

batch_size = 128
train_set = ImgDataset(test_x , label, train_transform)
val_set = train_set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
label_predictor = LabelPredictor().cuda()
#domain_classifier = DomainClassifier().cuda()
class_criterion = nn.CrossEntropyLoss()
domain_criterion = nn.BCEWithLogitsLoss()

feature_extractor = FeatureExtractor().cuda()
F_state_dict = torch.load('extractor_ada_model.bin')
feature_extractor.load_state_dict(F_state_dict)
label_predictor = LabelPredictor().cuda()
L_state_dict = torch.load('predictor_ada_model.bin')
label_predictor.load_state_dict(L_state_dict)

optimizer_F = optim.SGD(feature_extractor.parameters(), lr = 0.01, momentum=0.9)
optimizer_C =  optim.SGD(label_predictor.parameters(), lr = 0.01, momentum=0.9)

def train_epoch(train_loader):
    '''
      Args:
        source_dataloader: source data的dataloader
        target_dataloader: target data的dataloader
        lamb: 調控adversarial的loss係數。
    '''

    # D loss: Domain Classifier的loss
    # F loss: Feature Extrator & Label Predictor的loss
    # total_hit: 計算目前對了幾筆 total_num: 目前經過了幾筆
    running_D_loss, running_F_loss = 0.0, 0.0
    total_hit, total_num = 0.0, 0.0
    same_seeds(0)
    for i, data in enumerate(train_loader):
    
        source_data = data[0].cuda() #source_data.cuda()
        source_label = data[1].cuda() #source_label.cuda()
        
        #target_data = target_data.cuda()
        
        # 我們把source data和target data混在一起，否則batch_norm可能會算錯 (兩邊的data的mean/var不太一樣)
        #mixed_data = torch.cat([source_data, target_data], dim=0)
        #domain_label = torch.zeros([source_data.shape[0] + target_data.shape[0], 1]).cuda()
        # 設定source data的label為1
        #domain_label[:source_data.shape[0]] = 1

        # Step 1 : 訓練Domain Classifier
        feature = feature_extractor(source_data)
        # 因為我們在Step 1不需要訓練Feature Extractor，所以把feature detach避免loss backprop上去。
        #domain_logits = domain_classifier(feature.detach())
        #loss = domain_criterion(domain_logits, domain_label)
        #running_D_loss+= loss.item()
        #loss.backward()
        #optimizer_D.step()

        # Step 2 : 訓練Feature Extractor和Domain Classifier
        class_logits = label_predictor(feature[:source_data.shape[0]])
        #domain_logits = domain_classifier(feature)
        # loss為原本的class CE - lamb * domain BCE，相減的原因同GAN中的Discriminator中的G loss。
        loss = class_criterion(class_logits, source_label) #- lamb * domain_criterion(domain_logits, domain_label)
        running_F_loss+= loss.item()
        loss.backward()
        optimizer_F.step()
        optimizer_C.step()

        #optimizer_D.zero_grad()
        optimizer_F.zero_grad()
        optimizer_C.zero_grad()

        total_hit += torch.sum(torch.argmax(class_logits, dim=1) == source_label).item()
        total_num += source_data.shape[0]
        print(i, end='\r')

    return running_F_loss / (i+1), total_hit / total_num

# 訓練200 epochs
for epoch in range(400):
    train_F_loss, train_acc = train_epoch(train_loader)

    torch.save(feature_extractor.state_dict(), f'extractor_model_train.bin')
    torch.save(label_predictor.state_dict(), f'predictor_model_train.bin')

  
    print('epoch {:>3d}: train F loss: {:6.4f}, acc {:6.4f}'.format(epoch, train_F_loss, train_acc))

feature_extractor = FeatureExtractor().cuda()
F_state_dict = torch.load('extractor_model_train.bin')
feature_extractor.load_state_dict(F_state_dict)
label_predictor = LabelPredictor().cuda()
L_state_dict = torch.load('predictor_model_train.bin')
label_predictor.load_state_dict(L_state_dict)

same_seeds(0)
result = []
label_predictor.eval()
feature_extractor.eval()
for i, (test_data, _) in enumerate(test_dataloader):
    test_data = test_data.cuda()
    class_logits = label_predictor(feature_extractor(test_data))
    x = torch.argmax(class_logits, dim=1).cpu().detach().numpy()
    result.append(x)

import pandas as pd
result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv('prediction.csv',index=False)