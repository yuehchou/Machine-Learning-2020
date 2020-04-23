import os, cv2, time, torch, sys
import numpy as np
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from data_preprocessing import *
from training_model import *

load_model_path = None
workspace_dir = sys.argv[1]
results_fpath = sys.argv[2]

batch_size = 512

model = Classifier().cuda()
model = nn.DataParallel(model)

model.load_state_dict(torch.load(load_model_path))

# Predict
test_x = readfile(os.path.join(workspace_dir, "testing"), False)

print("Size of Testing data = {}".format(len(test_x)))

predict(model, batch_size, test_x, results_fpath)
print('Save results to {}'.format(results_fpath))
