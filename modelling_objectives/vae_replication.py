#%%
import time

import modelling_objectives_utils
import torch as t
from einops.layers.torch import Rearrange
from torch import nn
from tqdm.notebook import tqdm_notebook
import numpy as np
import plotly.express as px
from einops import rearrange
from typing import Tuple
import wandb

MAIN = __name__ == "__main__"

#%%
class Autoencoder(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, channels=1, size=28, mid_features=100, latent_dim_size=10):
        super().__init__()
        self.encoder = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(channels * size * size, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, channels * size * size),
            Rearrange("b (c h w) -> b c h w", h=size, w=size)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed
# %%
class ConvolutionalAutoencoder(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, channels=1, size=28, mid_channels=16, out_channels=32, mid_features=128, latent_dim_size=10, kernel_size=4, stride=2, padding=1):
        super().__init__()

        height = (size + 2 * padding - kernel_size) // stride + 1
        height = (height + 2 * padding - kernel_size) // stride + 1
        out_features = out_channels * height * height

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"), # nn.Flatten()
            nn.Linear(out_features, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, latent_dim_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, out_features),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c h w", h=height, w=height),
            nn.ConvTranspose2d(out_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed

#%%
def train_autoencoder(autoencoder: nn.Module, trainloader):
    epochs = 3

    optimiser = t.optim.Adam(autoencoder.parameters())

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)
        examples_seen = 0
        for (image, _) in progress_bar:
            optimiser.zero_grad()
            image.to(device)
            output = autoencoder(image)
            loss = nn.MSELoss()(output, image)
            loss.backward()
            optimiser.step()
            progress_bar.set_description(f"loss={loss:.3f}")
            examples_seen += len(image)
    modelling_objectives_utils.show_images(image.squeeze().detach(), rows=1, cols=5)
    modelling_objectives_utils.show_images(t.clip(output, -0.28/0.35, (1-0.28)/0.35).squeeze().detach(), rows=1, cols=5) # Need to clip to avoid min/max throwing px off

#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
latent_dim_size = 5
trainloader = modelling_objectives_utils.get_mnist(test=False)
autoencoder = ConvolutionalAutoencoder(latent_dim_size=latent_dim_size)
#train_autoencoder(autoencoder, trainloader)
# %%
class VariationalAutoencoder(nn.Module):
    encoder: nn.Sequential
    decoder: nn.Sequential

    def __init__(self, channels=1, size=28, mid_channels=16, out_channels=32, mid_features=128, latent_dim_size=10, kernel_size=4, stride=2, padding=1):
        super().__init__()

        height = (size + 2 * padding - kernel_size) // stride + 1
        height = (height + 2 * padding - kernel_size) // stride + 1
        out_features = out_channels * height * height

        self.encoder = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            Rearrange("b c h w -> b (c h w)"), # nn.Flatten()
            nn.Linear(out_features, mid_features),
            nn.ReLU()
        )
        self.linear_mu = nn.Linear(mid_features, latent_dim_size)
        self.linear_logsigma = nn.Linear(mid_features, latent_dim_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim_size, mid_features),
            nn.ReLU(),
            nn.Linear(mid_features, out_features),
            nn.ReLU(),
            Rearrange("b (c h w) -> b c h w", h=height, w=height),
            nn.ConvTranspose2d(out_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(mid_channels, channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x: t.Tensor):
        x = self.encoder(x)
        mu = self.linear_mu(x)
        logsigma = self.linear_logsigma(x)
        sigma = t.exp(logsigma)
        #x_compressed = t.normal(mu, sigma)
        x_compressed = t.randn_like(mu)
        x_compressed = mu + sigma * x_compressed
        
        x_reconstructed = self.decoder(x_compressed)
        return x_compressed, x_reconstructed, mu, logsigma
# %%
def train(autoencoder: nn.Module, trainloader, kl_loss_weight=0.1):
    epochs = 3

    optimiser = t.optim.Adam(autoencoder.parameters(), weight_decay=1e-5)

    for _ in range(epochs):
        progress_bar = tqdm_notebook(trainloader)
        examples_seen = 0
        for (image, _) in progress_bar:
            optimiser.zero_grad()
            image.to(device)
            _, output, mu, logsigma = autoencoder(image)
            reconstruction_loss = nn.MSELoss()(output, image)
            kl_loss = (- logsigma + 0.5 * (mu ** 2 + t.exp(2 * logsigma)) - 0.5).mean()

            loss = reconstruction_loss + kl_loss_weight * kl_loss
            loss.backward()
            optimiser.step()
            progress_bar.set_description(f"loss={loss:.3f}, r_loss={reconstruction_loss:.3f}, kl_loss={kl_loss:.3f}")
            examples_seen += len(image)
    modelling_objectives_utils.show_images(image.squeeze().detach(), rows=1, cols=5)
    modelling_objectives_utils.show_images(t.clip(output, -0.28/0.35, (1-0.28)/0.35).squeeze().detach(), rows=1, cols=5) # Need to clip to avoid min/max throwing px off#%%
#%%
device = t.device("cuda" if t.cuda.is_available() else "cpu")
latent_dim_size = 2
trainloader = modelling_objectives_utils.get_mnist(test=False)
variational_autoencoder = VariationalAutoencoder(latent_dim_size=latent_dim_size)
train(variational_autoencoder, trainloader)
# %%
def plot_latent_space(model, first_dimension=0, second_dimension=1):
    image, label = next(iter(trainloader))
    if isinstance(model, VariationalAutoencoder):
        z, _, _, _ = model(image.to(device))#.cpu().numpy()
    else:
        z = model.encoder(image.to(device))#.cpu().numpy()
    z = z.detach()
    fig = px.scatter(x=z[:, first_dimension], y=z[:, second_dimension], color=[str(x) for x in label.numpy()])
    fig.update_xaxes(title=f"Dimension {first_dimension}")
    fig.update_yaxes(title=f"Dimension {second_dimension}")
    fig.show()

def plot_all_latent_space(model, latent_dim_size):
    for i in range(latent_dim_size):
        for j in range(i + 1, latent_dim_size):
            plot_latent_space(model, i, j)
#%%
modelling_objectives_utils.plot_interpolation(variational_autoencoder, device, latent_dim_size)
plot_all_latent_space(variational_autoencoder, latent_dim_size)
# %%
