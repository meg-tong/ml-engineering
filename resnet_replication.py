#%%
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch as t
import torch.nn as nn
import torchvision
from PIL import Image

import arena_utils
from pytorch_replication import nn_replication


#%%
class ResidualBlock(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, first_stride=1):
        '''A single residual block with optional downsampling.

        For compatibility with the pretrained model, declare the left side branch first using a `Sequential`.

        If first_stride is > 1, this means the optional (conv + bn) should be present on the right branch. Declare it second using another `Sequential`.
        '''
        super().__init__()
        self.left = nn.Sequential(
            nn_replication.Conv2d(in_feats, out_feats, kernel_size=3, stride=first_stride, padding=1),
            nn_replication.BatchNorm2d(out_feats), 
            nn_replication.ReLU(), 
            nn_replication.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1), 
            nn_replication.BatchNorm2d(out_feats)
            )

        self.first_stride = first_stride
        self.right = nn.Sequential(
            nn_replication.Conv2d(in_feats, out_feats, kernel_size=1, stride=first_stride, padding=0),
            nn_replication.BatchNorm2d(out_feats)
        ) if self.first_stride > 1 else nn.Identity(oskar='cool', meg='more cool')

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.

        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x = self.left(x) + self.right(x)
        return nn.functional.relu(x)

# %%
class BlockGroup(nn.Module):
    def __init__(self, n_blocks: int, in_feats: int, out_feats: int, first_stride=1):
        '''An n_blocks-long sequence of ResidualBlock where only the first block uses the provided stride.'''
        super().__init__()
        self.resblock1 = ResidualBlock(in_feats, out_feats, first_stride=first_stride)
        self.resblocks = nn.Sequential(*[ResidualBlock(out_feats, out_feats, first_stride=1) for _ in range(n_blocks - 1)])

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Compute the forward pass.
        x: shape (batch, in_feats, height, width)

        Return: shape (batch, out_feats, height / first_stride, width / first_stride)
        '''
        x = self.resblock1(x)
        return self.resblocks(x)


# %%
class ResNet34(nn.Module):
    def __init__(
        self,
        n_blocks_per_group=[3, 4, 6, 3],
        out_features_per_group=[64, 128, 256, 512],
        first_strides_per_group=[1, 2, 2, 2],
        n_classes=1000,
    ):
        super().__init__()
        self.start = nn.Sequential(
            nn_replication.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn_replication.BatchNorm2d(64),
            nn_replication.ReLU(),
            nn_replication.MaxPool2d(kernel_size=3, stride=2)
        )

        in_features_per_group = [64] + out_features_per_group[:-1]
        self.block_groups = nn.Sequential(*[BlockGroup(n_blocks, in_feats, out_feats, first_stride) for n_blocks, in_feats, out_feats, first_stride in zip(n_blocks_per_group, in_features_per_group, out_features_per_group, first_strides_per_group)])

        self.end = nn.Sequential(
            nn_replication.AveragePool(),
            nn_replication.Flatten(),
            nn_replication.Linear(in_features=out_features_per_group[-1], out_features=n_classes)
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)

        Return: shape (batch, n_classes)
        '''
        x = self.start(x)
        x = self.block_groups(x)
        return self.end(x)
# %%
my_resnet = ResNet34()
pretrained_resnet = torchvision.models.resnet34(weights="DEFAULT")

arena_utils.print_param_count(my_resnet, pretrained_resnet)
# %%
def copy_weights(my_resnet: ResNet34, pretrained_resnet: torchvision.models.resnet.ResNet) -> ResNet34:
    '''Copy over the weights of `pretrained_resnet` to your resnet.'''

    mydict = my_resnet.state_dict()
    pretraineddict = pretrained_resnet.state_dict()

    # Check the number of params/buffers is correct
    assert len(mydict) == len(pretraineddict), "Number of layers is wrong. Have you done the prev step correctly?"

    # Initialise an empty dictionary to store the correct key-value pairs
    state_dict_to_load = {}

    for (mykey, myvalue), (pretrainedkey, pretrainedvalue) in zip(mydict.items(), pretraineddict.items()):
        state_dict_to_load[mykey] = pretrainedvalue

    my_resnet.load_state_dict(state_dict_to_load)

    return my_resnet

my_resnet = copy_weights(my_resnet, pretrained_resnet)
# %%
IMAGE_FILENAMES = [
    "chimpanzee.jpg",
    "golden_retriever.jpg",
    "platypus.jpg",
    "frogs.jpg",
    "fireworks.jpg",
    "astronaut.jpg",
    "iguana.jpg",
    "volcano.jpg",
    "goofy.jpg",
    "dragonfly.jpg",
]
IMAGE_FOLDER = Path("./data/resnet_inputs")
images = [Image.open(IMAGE_FOLDER / filename) for filename in IMAGE_FILENAMES]
# %%
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    torchvision.transforms.Resize((224, 224))
    ])

def prepare_data(images: List[Image.Image], transform) -> t.Tensor:
    '''
    Return: shape (batch=len(images), num_channels=3, height=224, width=224)
    '''
    return t.stack([transform(image) for image in images])

def predict(model, images):
    logits = model(images)
    predictions = logits.argmax(dim=1)
    return logits, predictions
# %%
with open("data/imagenet_labels.json") as f:
    imagenet_labels = list(json.load(f).values())
# %%

my_resnet.eval()
pretrained_resnet.eval()
with t.no_grad():
    prepared_images = prepare_data(images, transform)
    logits, predictions = predict(my_resnet, prepared_images)
    pretrained_logits, pretrained_predictions = predict(pretrained_resnet, prepared_images)
    for i, l, p, p_l, p_p in zip(images, logits, predictions, pretrained_logits, pretrained_predictions):
        plt.imshow(i)
        plt.show()
        print("Labels:", imagenet_labels[p.numpy()])
        print("Pretrained labels:", imagenet_labels[p_p.numpy()])
        print("Logits:", sorted(l.detach().numpy())[-5:], p.numpy())
        print("Pretrained logits:", sorted(p_l.detach().numpy())[-5:], p_p.numpy())
# %%
