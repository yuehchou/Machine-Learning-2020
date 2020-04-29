import torch
import torchvision.transforms as transforms
from torch.optim import Adam

def normalize(image):
  return (image - image.min()) / (image.max() - image.min())

layer_activations = None

def filter_explaination(x, model, cnnid, filterid, iteration=100, lr=1):
  model.eval()

  def hook(model, input, output):
    global layer_activations
    layer_activations = output

  hook_handle = model.cnn[cnnid].register_forward_hook(hook)

  model(x.cuda())

  filter_activations = layer_activations[:, filterid, :, :].detach().cpu()

  x = x.cuda()

  x.requires_grad_()
  optimizer = Adam([x], lr=lr)

  for iter in range(iteration):
    optimizer.zero_grad()
    model(x)
    objective = -layer_activations[:, filterid, :, :].sum()
    objective.backward()
    optimizer.step()

  filter_visualization = x.detach().cpu().squeeze()[0]
  hook_handle.remove()
  return filter_activations, filter_visualization
