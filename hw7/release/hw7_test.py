import os
import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import readfile
from utils import ImgDataset
from model import decode8
from model import StudentNet

load_model_path = 'model.pkl'
workspace_dir = sys.argv[1]
predict_fpath = sys.argv[2]
batch_size = 128

print("Reading data")
test_x = readfile(os.path.join(workspace_dir, "testing"), False)
print("Size of Testing data = {}".format(len(test_x)))

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
])

test_set = ImgDataset(test_x, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

params_de = decode8(load_model_path)
student_net = StudentNet(base=16).cuda()
student_net.load_state_dict(params_de)
model = student_net.cuda()

model.eval()
prediction = []
with torch.no_grad():
    for i, data in enumerate(test_loader):
        test_pred = model(data.cuda())
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        for y in test_label:
            prediction.append(y)

with open(predict_fpath, 'w') as f:
    f.write('Id,label\n')
    for i, y in  enumerate(prediction):
        f.write('{},{}\n'.format(i, y))
