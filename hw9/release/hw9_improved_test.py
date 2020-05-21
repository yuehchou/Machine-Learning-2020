import sys
import torch
import numpy as np
import torchvision.transforms as transforms
from model import AE_improved
from clustering import inference_improved
from clustering import predict_improved
from clustering import invert
from clustering import save_prediction

trainX_path = sys.argv[1]
checkpoints_path = sys.argv[2]
prediction_path = sys.argv[3]

trainX = np.load(trainX_path)

test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5)),
])

# load model
model = AE_improved().cuda()
model.load_state_dict(torch.load(checkpoints_path))
model.eval()

latents = inference_improved(X=trainX, test_transform=test_transform, model=model)
pred, X_embedded = predict_improved(latents)
save_prediction(invert(pred), prediction_path)
# save_prediction(pred, prediction_path)
