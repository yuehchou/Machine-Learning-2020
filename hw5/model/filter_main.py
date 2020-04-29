import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import *
from filter_explain import *
from model import *

workspace_dir = './../data'
save_filter_fpath = './../fig/filter_layer_15.png'
save_fpath = './../fig/filter_layer_15_explaination_results.png'
load_checkpoint = './../../hw3/model/best_checkpoint_norm_cnn.pt'

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
# model = nn.DataParallel(model)
model.load_state_dict(torch.load(load_checkpoint), strict=False)

for data in train_loader:
    images = data[0]
    labels = data[1]
    break

filter_activations, filter_visualization = filter_explaination(images, model, cnnid=15, filterid=0, iteration=100, lr=0.1)

# plt.imshow(normalize(filter_visualization.permute(1, 2, 0)))
plt.imshow(filter_visualization.permute(1, 2, 0))
plt.show()
plt.savefig(save_filter_fpath)
plt.close()

fig, axs = plt.subplots(len(train_x), 2, figsize=(8, 24))
for i, img in enumerate(images):
    axs[i][0].imshow(invTrans(img).permute(1, 2, 0))
    axs[i][0].set_title(str(i))
for i, img in enumerate(filter_activations):
    axs[i][1].imshow(img)
    # axs[i][1].imshow(normalize(img))
    axs[i][1].set_title(str(i))

plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                wspace=None, hspace=0.5)
plt.show()
plt.savefig(save_fpath)
plt.close()
