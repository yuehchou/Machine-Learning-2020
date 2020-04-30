import os, sys
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from skimage.segmentation import slic
from torch.utils.data import DataLoader
from data_preprocessing import *
from model import *
from lime import lime_image

def predict(input):
    # input: numpy array, (batches, height, width, channels)

    model = Classifier().cuda()
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(load_checkpoint), strict=False)

    model.eval()
    input = torch.FloatTensor(input).permute(0, 3, 1, 2)

    output = model(input.cuda())
    return output.detach().cpu().numpy()

def segmentation(input):
    return slic(input, n_segments=100, compactness=1, sigma=1)


workspace_dir = os.path.join(sys.argv[1])
save_fpath = os.path.join(sys.argv[2], 'lime.png')
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
])

invTrans = transforms.Compose([
    transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),
])

train_set = ImgDataset(train_x, train_y, train_transform)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

for data in train_loader:
    images = data[0]
    labels = data[1]
    break

image_list = [img for img in images]

fig, axs = plt.subplots(1, len(train_x), figsize=(24, 6))
for idx, (image, label) in enumerate(zip(images.permute(0, 2, 3, 1).numpy(), labels)):
    x = image.astype(np.double)

    explainer = lime_image.LimeImageExplainer()
    explaination = explainer.explain_instance(image=x, classifier_fn=predict, segmentation_fn=segmentation)

    lime_img, mask = explaination.get_image_and_mask(
                                label=explaination.top_labels[0],
                                positive_only=False,
                                hide_rest=False,
                                num_features=11,
                                min_weight=0.05
                            )
    axs[idx].imshow(lime_img)
    axs[idx].set_title(str(label.item()))

plt.show()
plt.savefig(save_fpath)
plt.close()
