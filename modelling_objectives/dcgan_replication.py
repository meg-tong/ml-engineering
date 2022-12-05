#%%
import torch as t
from typing import Union
from torch import nn
import torch.nn.functional as F
import plotly.express as px
import plotly.graph_objects as go
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from fancy_einsum import einsum
import os
from tqdm.auto import tqdm
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader, TensorDataset
import wandb
import obj_utils
MAIN = __name__ == "__main__"
# %%
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int = 100,         # size of the random vector we use for generating outputs
        img_size: int = 64,                 # size of the images we're generating
        img_channels: int = 3,              # indicates RGB images
        generator_num_features: int = 1024, # number of channels after first projection and reshaping
        n_layers: int = 4,                  # number of CONV_n layers
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        size = img_size // (2 ** n_layers)

        self.project_and_reshape = nn.Sequential(
            nn.Linear(latent_dim_size, size * size * generator_num_features, bias=False),
            Rearrange("b (c h w) -> b c h w", h=size, w=size),
            nn.BatchNorm2d(generator_num_features),
            nn.ReLU()
        )

        tconv_layers = []
        out_channel = generator_num_features
        for i in range(n_layers):
            in_channel = generator_num_features // (2 ** i)
            out_channel = generator_num_features // (2 ** (i + 1)) if i < n_layers - 1 else img_channels
            tconv_layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding))
            if i < n_layers - 1:
                tconv_layers.append(nn.BatchNorm2d(out_channel))
                tconv_layers.append(nn.ReLU())
            else:
                tconv_layers.append(nn.Tanh())
        self.tconv_layers = nn.Sequential(*tconv_layers)

    def forward(self, x: t.Tensor):
        x = self.project_and_reshape(x)
        x = self.tconv_layers(x)
        return x

class Discriminator(nn.Module):

    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        generator_num_features: int = 1024,
        n_layers: int = 4,                  
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()

        conv_layers = []
        for i in range(n_layers):
            in_channel = generator_num_features // (2 ** (n_layers - (i))) if i > 0 else img_channels
            out_channel = generator_num_features // (2 **(n_layers - (i + 1)))

            conv_layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding))
            if i > 0:
                conv_layers.append(nn.BatchNorm2d(out_channel))
            conv_layers.append(nn.LeakyReLU())
        self.conv_layers = nn.Sequential(*conv_layers)

        size = img_size // (2 ** n_layers)
        self.reshape_and_project = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(size * size * generator_num_features, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: t.Tensor):
        
        x = self.conv_layers(x)
        x = self.reshape_and_project(x)
        return x


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,         # size of the random vector we use for generating outputs
        img_size: int = 64,                 # size of the images we're generating
        img_channels: int = 3,              # indicates RGB images
        generator_num_features: int = 1024, # number of channels after first projection and reshaping
        n_layers: int = 4,                  # number of CONV_n layers
        kernel_size: int = 4,
        stride: int = 2,
        padding: int = 1
    ):
        super().__init__()
        self.netG = Generator(
            latent_dim_size=latent_dim_size,
            img_size=img_size,
            img_channels=img_channels,
            generator_num_features=generator_num_features,
            n_layers=n_layers,
            kernel_size=kernel_size,
            stride=stride
        )
        self.netD = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
            generator_num_features=generator_num_features,
            n_layers=n_layers
        )

#%%
if MAIN:
    dcgan = DCGAN()
# %%
