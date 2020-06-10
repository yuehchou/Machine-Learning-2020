import os
import sys
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import same_seeds
from model import FeatureExtractor, LabelPredictor

data_folder = sys.argv[1]
prediction_path = sys.argv[2]

F_state_dict = torch.load('./extractor_model.bin')
L_state_dict = torch.load('./predictor_model.bin')

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
])

target_dataset = ImageFolder(os.path.join(data_folder, 'test_data'), transform=target_transform)
test_dataloader = DataLoader(target_dataset, batch_size=128*4, shuffle=False)

feature_extractor = FeatureExtractor().cuda()
feature_extractor.load_state_dict(F_state_dict)
label_predictor = LabelPredictor().cuda()
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

result = np.concatenate(result)

# Generate your submission
df = pd.DataFrame({'id': np.arange(0,len(result)), 'label': result})
df.to_csv(prediction_path,index=False)
