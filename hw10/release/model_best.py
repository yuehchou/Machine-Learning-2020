import torch
from torch import nn
import torch.nn.functional as F

class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(32 * 32 * 3, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 12),
            nn.ReLU(True),
            nn.Linear(12, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 12),
            nn.ReLU(True),
            nn.Linear(12, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 32 * 32 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 36, 3, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(p=0.4, inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(36, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class conv_autoencoder1(nn.Module):
    def __init__(self):
        super(conv_autoencoder1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),            # [batch, 12, 16, 16]
            nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 36, 3, stride=1, padding=1),           # [batch, 24, 8, 8]
            nn.BatchNorm2d(36, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.Dropout(p=0.4, inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(36, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(True),
            nn.MaxPool2d(2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1) # out: c x 14 x 14
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.conv3 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1) # out: c x 7 x 7
        self.fc21 = nn.Linear(48*4*4, 200*4)
        self.fc22 = nn.Linear(48*4*4, 200*4)
        self.fc3 = nn.Linear(200*4, 48*4*4)
        self.dconv3 = nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1)
        self.dconv2 = nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1)
        self.dconv1 = nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=1, padding=1),
            nn.BatchNorm2d(12, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.MaxPool2d(2),
            nn.Conv2d(12, 24, 3, stride=1, padding=1),
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.2, inplace=False),
            nn.MaxPool2d(2),

            nn.Conv2d(24, 24, 3, stride=1, padding=1),
            nn.BatchNorm2d(24, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.4, inplace=False),

            nn.Conv2d(24, 48, 3, stride=1, padding=1),
            nn.BatchNorm2d(48, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            nn.Dropout(p=0.3, inplace=False),
            nn.MaxPool2d(2)
        )

    def encode(self, x):
        x = F.relu(self.encoder(x))
        x = x.view(x.size(0), -1) # flatten batch of multi-channel feature maps to a batch of feature vectors
        h1 = x
        return self.fc21(h1), self.fc22(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        x = h3
        x = x.view(x.size(0), 48, 4, 4)
        x = F.relu(self.dconv3(x))
        x = F.relu(self.dconv2(x))
        x = torch.tanh(self.dconv1(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar
