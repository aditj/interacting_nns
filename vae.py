
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.latent_size = latent_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, latent_size * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_size),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    def loss(self, x, x_recon, mu, logvar):
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_div
    
    def sample(self, num_samples):
        z = torch.randn(num_samples, self.latent_size)
        return self.decode(z)

### Sample data to learn distribution of 
### (x, y) pairs where x is a 2D vector and
### y is a scalar

import numpy as np

x = np.random.randn(1000, 2)

y = np.random.randn(1000, 1) + x[:, 0:1] + x[:, 1:2]
y = y.reshape(-1, 1)

### Train VAE

vae = VAE(3, 10, 2)
optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

for epoch in range(1000):
    x_recon, mu, logvar = vae(torch.Tensor(np.hstack([x, y])))
    loss = vae.loss(torch.Tensor(np.hstack([x, y])), x_recon, mu, logvar)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch, loss.item()))

### Sample from VAE

x_recon, mu, logvar = vae(torch.Tensor(np.hstack([x, y])))
z = vae.reparameterize(mu, logvar)
x_recon = vae.decode(z)
x_recon = x_recon.detach().numpy()

import matplotlib.pyplot as plt
### Plot samples
plt.scatter(x[:, 0], x[:, 1], c='b', alpha=0.5)

plt.scatter(x_recon[:, 0], x_recon[:, 1], c='r', alpha=0.5)
plt.savefig("vae.png")
