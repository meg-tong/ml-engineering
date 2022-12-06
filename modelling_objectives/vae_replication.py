#%%
import time

import modelling_objectives_utils
import torch as t
from einops.layers.torch import Rearrange
from torch import nn
from tqdm.notebook import tqdm_notebook

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
            nn.Conv2d(in_channels=channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
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
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=mid_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=mid_channels, out_channels=channels, kernel_size=kernel_size, stride=stride, padding=padding)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        x_compressed = self.encoder(x)
        x_reconstructed = self.decoder(x_compressed)
        return x_reconstructed

#%%
def train(autoencoder: nn.Module):
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    epochs = 2

    trainloader = modelling_objectives_utils.get_mnist(test=False)
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
#train(Autoencoder())
# %%
train(ConvolutionalAutoencoder())
# %%
