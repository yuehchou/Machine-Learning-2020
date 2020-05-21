import sys
import torch
import numpy as np
from model import AE_baseline
from clustering import inference
from clustering import predict
from clustering import invert
from clustering import save_prediction

trainX_path = sys.argv[1]
checkpoints_path = sys.argv[2]
prediction_path = sys.argv[3]

trainX = np.load(trainX_path)

# load model
model = AE_baseline().cuda()
model.load_state_dict(torch.load(checkpoints_path))
model.eval()

latents = inference(X=trainX, model=model)
pred, X_embedded = predict(latents)

save_prediction(invert(pred), prediction_path)
# save_prediction(pred, prediction_path)
