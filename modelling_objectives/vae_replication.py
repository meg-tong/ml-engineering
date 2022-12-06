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
def train():
    
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    epochs = 3

    trainloader = modelling_objectives_utils.get_mnist(test=False)
    autoencoder = Autoencoder()
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
    modelling_objectives_utils.show_images(t.clip(output, -0.28/0.35, (1-0.28)/0.35).squeeze().detach(), rows=1, cols=5)
        
train()
# %%
# %%
