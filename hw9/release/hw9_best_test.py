import sys
import torch
import numpy as np
from model import AE_best
from clustering import inference_best
from clustering import predict_best
from clustering import invert
from clustering import save_prediction

trainX_path = sys.argv[1]
checkpoints_path = sys.argv[2]
prediction_path = sys.argv[3]

trainX = np.load(trainX_path)

# load model
model = AE_best().cuda()
model.load_state_dict(torch.load(checkpoints_path))
model.eval()

latents = inference_best(X=trainX, model=model)
pred, X_embedded = predict_best(latents)

# save_prediction(invert(pred), prediction_path)
save_prediction(pred, prediction_path)
