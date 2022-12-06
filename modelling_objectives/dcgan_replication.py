#%%
import time
from typing import Tuple, Optional

import numpy as np
import obj_utils
import torch as t
from einops.layers.torch import Rearrange
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm.notebook import tqdm_notebook

import wandb

MAIN = __name__ == "__main__"
# %%
class Generator(nn.Module):

    def __init__(
        self,
        latent_dim_size: int,        # size of the random vector we use for generating outputs
        img_size: int,               # size of the images we're generating
        img_channels: int,           # indicates RGB images
        generator_num_features: int, # number of channels after first projection and reshaping
        n_layers: int,               # number of CONV_n layers
        kernel_size: int,
        stride: int,
        padding: int
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
            tconv_layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
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
        img_size,
        img_channels,
        generator_num_features,
        n_layers,                  
        kernel_size,
        stride,
        padding
    ):
        super().__init__()

        conv_layers = []
        for i in range(n_layers):
            in_channel = generator_num_features // (2 ** (n_layers - (i))) if i > 0 else img_channels
            out_channel = generator_num_features // (2 **(n_layers - (i + 1)))

            conv_layers.append(nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, bias=False))
            if i > 0:
                conv_layers.append(nn.BatchNorm2d(out_channel))
            conv_layers.append(nn.LeakyReLU(negative_slope=0.2))
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

class DCGANArgs():
    latent_dim_size: int = 100
    img_size: int = 64
    img_channels: int = 3
    generator_num_features: int = 512
    n_layers: int = 4
    trainset: datasets.ImageFolder
    lr: float = 0.0002
    betas: Tuple[float, float] = (0.5, 0.999)
    batch_size: int = 8
    epochs: int = 1
    kernel_size: int = 4
    stride: int = 2
    padding: int = 1
    track: bool = True
    cuda: bool = True
    seconds_between_image_logs: int = 40

class DCGAN(nn.Module):
    discriminator: Discriminator
    generator: Generator

    def __init__(
        self,
        latent_dim_size: int,        # size of the random vector we use for generating outputs
        img_size: int,               # size of the images we're generating
        img_channels: int,           # indicates RGB images
        generator_num_features: int, # number of channels after first projection and reshaping
        n_layers: int,               # number of CONV_n layers
        kernel_size: int,
        stride: int,
        padding: int
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        
        self.generator = Generator(
            latent_dim_size=latent_dim_size,
            img_size=img_size,
            img_channels=img_channels,
            generator_num_features=generator_num_features,
            n_layers=n_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.discriminator = Discriminator(
            img_size=img_size,
            img_channels=img_channels,
            generator_num_features=generator_num_features,
            n_layers=n_layers,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

    @staticmethod
    def from_args(args: DCGANArgs):
        return DCGAN(
            latent_dim_size=args.latent_dim_size,
            img_size=args.img_size,
            img_channels=args.img_channels,
            generator_num_features=args.generator_num_features,
            n_layers=args.n_layers,
            kernel_size=args.kernel_size,
            stride=args.stride,
            padding=args.padding
        )

#%%
import w5d1_solutions
args = DCGANArgs()
dcgan = DCGAN.from_args(DCGANArgs())
obj_utils.print_param_count(dcgan.discriminator, w5d1_solutions.celeb_mini_DCGAN.netD)
print(w5d1_solutions.celeb_mini_DCGAN.netD)

#%%
def initialize_weights(model, mean=0, batch_norm_mean=1, std=0.02) -> None:
    for module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            nn.init.normal_(module.weight, mean=batch_norm_mean, std=std)
            nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=mean, std=std)
#%%
def generate_dataset(folder="../data/img_align_celeba", img_size=64):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    trainset = ImageFolder(
        root=folder,
        transform=transform
    )
    return trainset

if MAIN:
    obj_utils.show_images(generate_dataset(), rows=3, cols=5)
#%%

def train(args: DCGANArgs):#Optional[DCGANArgs] = None):
    #TODO: wandb tracking

    device = t.device("cuda" if args.cuda and t.cuda.is_available() else "cpu")
    dcgan = DCGAN.from_args(args).to(device).train()
    initialize_weights(dcgan.discriminator)
    initialize_weights(dcgan.generator)
    
    #paper_batch_size = 128
    #assert paper_batch_size % args.batch_size == 0
    #batch_size_ratio = paper_batch_size // args.batch_size
    generator_optimiser = t.optim.Adam(dcgan.generator.parameters(), lr=args.lr, betas=args.betas) 
    discriminator_optimiser = t.optim.Adam(dcgan.discriminator.parameters(), lr=args.lr, betas=args.betas)
    #TODO: check the hyperparameters are the same?

    trainset = generate_dataset()
    trainloader = DataLoader(trainset, shuffle=True, batch_size=args.batch_size)
    
    for epoch in range(args.epochs):


        progress_bar = tqdm_notebook(trainloader)
        images_seen = 0
        stopwatch = time.time()
        for (real_image, _) in progress_bar:
            images_seen += len(real_image)

            noise = t.randn((args.batch_size, dcgan.latent_dim_size), device=device)
            fake_image = dcgan.generator(noise)

            # Training the discriminator
            real_image = real_image.to(device)
            real_image_loss = t.log(dcgan.discriminator(real_image)).mean()
            fake_image_loss = t.log(1 - dcgan.discriminator(fake_image.detach())).mean()
            discriminator_loss = - (real_image_loss + fake_image_loss) #TODO: check whether this should aggregate across 'mega'batches?

            #if images_seen % paper_batch_size == 0:
            dcgan.discriminator.zero_grad()
            discriminator_loss.backward() 
            discriminator_optimiser.step()

            # Training the generator
            generator_loss = - t.log(dcgan.discriminator(fake_image)).mean()

            dcgan.discriminator.zero_grad()
            dcgan.generator.zero_grad()
            generator_loss.backward()
            generator_optimiser.step()
            
            # Logging
            progress_bar.set_description(f"Epoch={epoch}, discriminator_loss={discriminator_loss:.3f}, generator_loss={generator_loss:.3f}, real_image_loss={real_image_loss:.3f}, fake_image_loss={fake_image_loss:.3f}")
            if time.time() - stopwatch > args.seconds_between_image_logs:
                obj_utils.show_images(fake_image.detach(), rows=2, cols=4)
                stopwatch = time.time()
            
t.autograd.set_detect_anomaly(True)

if MAIN:
    args = DCGANArgs()
    args.seconds_between_image_logs = 10
    train(DCGANArgs())
# %%
