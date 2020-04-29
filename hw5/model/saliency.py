import torch
import torch.nn as nn


def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

def compute_saliency_maps(x, y, model):
  model.eval()
  x = x.cuda()

  x.requires_grad_()

  y_pred = model(x)

  loss_func = torch.nn.CrossEntropyLoss()
  loss = loss_func(y_pred, y.cuda())
  loss.backward()

  saliencies = x.grad.abs().detach().cpu()
  saliencies = torch.stack([normalize(item) for item in saliencies])
  return saliencies
