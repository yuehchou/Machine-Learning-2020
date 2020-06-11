import sys
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils import same_seeds, get_dataset
from model import Generator, Discriminator

img_dir = sys.argv[1]
checkpoint_fpath = sys.argv[2]

# hyperparameters 
batch_size = 64
z_dim = 100
lr = 1e-4
# n_epoch = 1
n_epoch = 20
same_seeds(0)

# model
G = Generator(in_dim=z_dim).cuda()
D = Discriminator(3).cuda()
G.train()
D.train()

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

dataset = get_dataset(img_dir)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

# for logging
z_sample = Variable(torch.randn(100, z_dim)).cuda()

for e, epoch in enumerate(range(n_epoch)):
    for i, data in enumerate(dataloader):
        imgs = data
        imgs = imgs.cuda()

        bs = imgs.size(0)

        """ Train D """
        z = Variable(torch.randn(bs, z_dim)).cuda()
        r_imgs = Variable(imgs).cuda()
        f_imgs = G(z)

        # label        
        r_label = torch.ones((bs)).cuda()
        f_label = torch.zeros((bs)).cuda()

        # dis
        r_logit = D(r_imgs.detach())
        f_logit = D(f_imgs.detach())

        # compute loss
        loss_D = -torch.mean(r_logit) + torch.mean(f_logit)

        # update model
        D.zero_grad()
        loss_D.backward()
        opt_D.step()
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)

        """ train G """
        # leaf
        z = Variable(torch.randn(bs, z_dim)).cuda()
        f_imgs = G(z)

        # dis
        f_logit = D(f_imgs)

        # compute loss
        loss_G = -torch.mean(f_logit)

        # update model
        G.zero_grad()
        loss_G.backward()
        opt_G.step()

        # log
        print(f'\rEpoch [{epoch+1}/{n_epoch}] {i+1}/{len(dataloader)} Loss_D: {loss_D.item():.4f} Loss_G: {loss_G.item():.4f}', end='')
    G.eval()
    f_imgs_sample = (G(z_sample).data + 1) / 2.0
    G.train()

# save checkpoints
torch.save(G.state_dict(), checkpoint_fpath)
