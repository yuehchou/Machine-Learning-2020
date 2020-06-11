import sys
import torch
import torchvision
import matplotlib.pyplot as plt
from model import Generator

checkpoint_fpath = sys.argv[1]
img_fpath = sys.argv[2]
z_sample = torch.load('./z_sample_wg.pt')

z_dim = 100
n_output = 20

# load pretrained model
G = Generator(z_dim)
G.load_state_dict(torch.load(checkpoint_fpath))
G.eval()
G.cuda()

# generate images and save the result
imgs_sample = (G(z_sample).data + 1) / 2.0
torchvision.utils.save_image(imgs_sample, img_fpath, nrow=10)
