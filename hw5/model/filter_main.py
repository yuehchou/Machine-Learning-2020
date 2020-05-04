import torch, os, sys
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import *
from filter_explain import *
from model import *

workspace_dir = sys.argv[1]
load_checkpoint = None

save_filter_fpath_6 = os.path.join(sys.argv[2], 'filter_layer_6.png')
save_fpath_6 = os.path.join(sys.argv[2], 'filter_layer_6_explaination_results.png')
save_filter_fpath_9 = os.path.join(sys.argv[2], 'filter_layer_9.png')
save_fpath_9 = os.path.join(sys.argv[2], 'filter_layer_9_explaination_results.png')
save_filter_fpath_12 = os.path.join(sys.argv[2], 'filter_layer_12.png')
save_fpath_12 = os.path.join(sys.argv[2], 'filter_layer_12_explaination_results.png')
save_filter_fpath_15 = os.path.join(sys.argv[2], 'filter_layer_15.png')
save_fpath_15 = os.path.join(sys.argv[2], 'filter_layer_15_explaination_results.png')

batch_size = 256

print("Reading data")
train_x, train_y = readfile(workspace_dir, True)
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
# model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_checkpoint), strict=False)

for data in train_loader:
    images = data[0]
    labels = data[1]
    break

filter_activations_6, filter_visualization_6 = filter_explaination(images, model, cnnid=6, filterid=0, iteration=75, lr=0.1)
filter_activations_9, filter_visualization_9 = filter_explaination(images, model, cnnid=9, filterid=0, iteration=75, lr=0.1)
filter_activations_12, filter_visualization_12 = filter_explaination(images, model, cnnid=12, filterid=0, iteration=75, lr=0.1)
filter_activations_15, filter_visualization_15 = filter_explaination(images, model, cnnid=15, filterid=0, iteration=75, lr=0.1)

plt.imshow(filter_visualization_6.permute(1, 2, 0))
plt.show()
plt.savefig(save_filter_fpath_6)
plt.close()

fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for i, img in enumerate(images):
    axs[i][0].imshow(invTrans(img).permute(1, 2, 0))
    axs[i][0].set_title(str(i))
for i, img in enumerate(filter_activations_6):
    axs[i][1].imshow(img)
    axs[i][1].set_title(str(i))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath_6)
plt.close()

plt.imshow(filter_visualization_9.permute(1, 2, 0))
plt.show()
plt.savefig(save_filter_fpath_9)
plt.close()

fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for i, img in enumerate(images):
    axs[i][0].imshow(invTrans(img).permute(1, 2, 0))
    axs[i][0].set_title(str(i))
for i, img in enumerate(filter_activations_9):
    axs[i][1].imshow(img)
    axs[i][1].set_title(str(i))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath_9)
plt.close()

plt.imshow(filter_visualization_12.permute(1, 2, 0))
plt.show()
plt.savefig(save_filter_fpath_12)
plt.close()

fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for i, img in enumerate(images):
    axs[i][0].imshow(invTrans(img).permute(1, 2, 0))
    axs[i][0].set_title(str(i))
for i, img in enumerate(filter_activations_12):
    axs[i][1].imshow(img)
    axs[i][1].set_title(str(i))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath_12)
plt.close()

plt.imshow(filter_visualization_15.permute(1, 2, 0))
plt.show()
plt.savefig(save_filter_fpath_15)
plt.close()

fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for i, img in enumerate(images):
    axs[i][0].imshow(invTrans(img).permute(1, 2, 0))
    axs[i][0].set_title(str(i))
for i, img in enumerate(filter_activations_15):
    axs[i][1].imshow(img)
    axs[i][1].set_title(str(i))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath_15)
plt.close()
