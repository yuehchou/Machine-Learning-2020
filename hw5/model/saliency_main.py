import os, sys, torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import *
from saliency import *
from model import *

workspace_dir = os.path.join(sys.argv[1])
save_fpath = os.path.join(sys.argv[2], 'saliency_results.png')
load_checkpoint = './best_checkpoint_norm_cnn.pt'

batch_size = 256

print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.RandomResizedCrop(128),
    transforms.ToTensor(),
    normalize,
])

invTrans = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
])

train_set = ImgDataset(train_x, train_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

model = Classifier().cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_checkpoint), strict=False)

for data in train_loader:
    images = data[0]
    labels = data[1]
    break

saliencies = compute_saliency_maps(images, labels, model)

# count = 0
fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for row, target in enumerate([images, saliencies]):
    count = 0
    for column, img in enumerate(target):
        if row == 0:
            temp = invTrans(img).permute(1,2,0).numpy()
        else:
            temp = img.permute(1,2,0).numpy()
        axs[column][row].imshow(temp)
        axs[column][row].set_title(str(count))
        count += 1

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath)
plt.close()
