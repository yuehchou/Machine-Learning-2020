import sys
import random
import torch
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from utils import same_seeds
from model import conv_autoencoder

# ==========
# Setting
batch_size = 128
model_type = 'cnn'
# ==========

test_path = sys.argv[1]
model_path = sys.argv[2]
prediction_path = sys.argv[3]

test = np.load(test_path, allow_pickle=True)
model = conv_autoencoder().cuda()

if model_type == 'fcn' or model_type == 'vae':
    y = test.reshape(len(test), -1)
else:
    y = test

data = torch.tensor(y, dtype=torch.float)
test_dataset = TensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

model = torch.load(model_path)

model.eval()
reconstructed = list()
for i, data in enumerate(test_dataloader):
    if model_type == 'cnn':
        img = data[0].transpose(3, 1).cuda()
    else:
        img = data[0].cuda()
    output = model(img)
    if model_type == 'cnn':
        output = output.transpose(3, 1)
    elif model_type == 'vae':
        output = output[0]
    reconstructed.append(output.cpu().detach().numpy())

reconstructed = np.concatenate(reconstructed, axis=0)
anomality = np.sqrt(np.sum(np.square(reconstructed - y).reshape(len(y), -1), axis=1))
y_pred = anomality

with open(prediction_path, 'w') as f:
    f.write('id,anomaly\n')
    for i in range(len(y_pred)):
        f.write('{},{}\n'.format(i+1, y_pred[i]))
