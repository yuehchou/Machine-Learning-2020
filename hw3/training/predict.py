import os, cv2, time, torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import *
from training_model import *

workspace_dir = './../data/food-11'
# save_model_path = './../model/checkpoint_self_pretrained_cnn_template.pt'
save_model_path = './../model/checkpoint.pt'
# results_fpath = './../results/predict_self_pretrained_cnn_template.csv'
results_fpath = './../results/predict_best_cnn.csv'

# batch_size = 128
batch_size = 512

model = Classifier().cuda()
model = nn.DataParallel(model)
# model = models.resnet152().cuda()

model.load_state_dict(torch.load(save_model_path))

# Predict
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
# test_x = readfile_resnet(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

predict(model, batch_size, test_x, results_fpath)
print('Save results to {}'.format(results_fpath))
