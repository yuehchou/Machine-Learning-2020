import shap, sys, os, torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from data_preprocessing import *
from model import *

workspace_dir = os.path.join(sys.argv[1])
save_fpath = os.path.join(sys.argv[2], 'shap_results.png')
load_checkpoint = './best_checkpoint_norm_cnn.pt'

batch_size = 256

print("Reading data")
train_x, train_y = readfile(os.path.join(workspace_dir, "training"), True)
print("Size of training data = {}".format(len(train_x)))

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(150),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])

test_set = ImgDataset(train_x, train_y, test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = Classifier().cuda()
model.load_state_dict(torch.load(load_checkpoint), strict=False)

for data in test_loader:
    test_images = data[0]
    break

e = shap.DeepExplainer(model, test_images.cuda())
shap_values = e.shap_values(test_images)

shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)


shap.image_plot(shap_numpy, test_numpy)
plt.savefig(save_fpath)
plt.close()
