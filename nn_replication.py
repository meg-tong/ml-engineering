# %%
import torch as t
import torch.nn as nn
from einops import repeat
from fancy_einsum import einsum
import numpy as np
import typing
from typing import Optional

import importlib
import utils_w0d2
import utils_w0d3
# %%
def conv1d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv1d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    out_channels, in_channels, kernel_width = weights.shape
    batch, in_channels, width = x.shape
    x_exp = x.as_strided(
        size=(batch, in_channels, kernel_width, width - kernel_width + 1),
        stride=(x.stride()[0], x.stride()[1], x.stride()[-1], x.stride()[-1])
    )

    return einsum('b in_ch kernel_w new_width, o_ch in_ch kernel_w -> b o_ch new_width', x_exp, weights)

utils_w0d2.test_conv1d_minimal(conv1d_minimal)
# %%
def conv2d_minimal(x: t.Tensor, weights: t.Tensor) -> t.Tensor:
    '''Like torch's conv2d using bias=False and all other keyword arguments left at their default values.

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)

    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    batch, in_channels, height, width = x.shape
    x_exp = x.as_strided(
        size=(batch, in_channels, kernel_height, height - kernel_height + 1, kernel_width, width - kernel_width + 1),
        stride=(x.stride()[0], x.stride()[1], x.stride()[2], x.stride()[2], x.stride()[3], x.stride()[3])
    )

    return einsum('b in_ch kernel_h h kernel_w w, o_ch in_ch kernel_h kernel_w -> b o_ch h w', x_exp, weights)

utils_w0d2.test_conv2d_minimal(conv2d_minimal)
# %%
def pad1d(x: t.Tensor, left: int, right: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, width), dtype float32

    Return: shape (batch, in_channels, left + right + width)
    '''
    padded_x = x.new_full((x.shape[0], x.shape[1], left + right + x.shape[2]), pad_value)
    padded_x[..., left:left + x.shape[2]] = x
    return padded_x

utils_w0d2.test_pad1d(pad1d)
utils_w0d2.test_pad1d_multi_channel(pad1d)
# %%
def pad2d(x: t.Tensor, left: int, right: int, top: int, bottom: int, pad_value: float) -> t.Tensor:
    '''Return a new tensor with padding applied to the edges.

    x: shape (batch, in_channels, height, width), dtype float32

    Return: shape (batch, in_channels, top + height + bottom, left + width + right)
    '''
    padded_x = x.new_full((x.shape[0], x.shape[1], top + bottom + x.shape[2], left + right + x.shape[3]), pad_value)
    padded_x[..., top:top + x.shape[2], left:left + x.shape[3]] = x
    return padded_x

utils_w0d2.test_pad2d(pad2d)
utils_w0d2.test_pad2d_multi_channel(pad2d)

# %%
def conv1d(x, weights, stride: int = 1, padding: int = 0) -> t.Tensor:
    '''Like torch's conv1d using bias=False.

    x: shape (batch, in_channels, width)
    weights: shape (out_channels, in_channels, kernel_width)

    Returns: shape (batch, out_channels, output_width)
    '''
    batch, in_channels, width = x.shape
    out_channels, in_channels, kernel_width = weights.shape

    x_padded = pad1d(x, padding, padding, 0.0)
    x_exp = x_padded.as_strided(
        size=(batch, in_channels, kernel_width, int(np.floor((width + 2 * padding - kernel_width) / stride)) + 1),
        stride=(x_padded.stride()[0], x_padded.stride()[1], x_padded.stride()[-1], x_padded.stride()[-1] * stride)
    )

    return einsum('b in_ch kernel_w new_width, o_ch in_ch kernel_w -> b o_ch new_width', x_exp, weights)

utils_w0d2.test_conv1d(conv1d)
# %%
IntOrPair = typing.Union[int, typing.Tuple[int, int]]
Pair = typing.Tuple[int, int]

def force_pair(v: IntOrPair) -> Pair:
    '''Convert v to a pair of int, if it isn't already.'''
    if isinstance(v, tuple):
        if len(v) != 2:
            raise ValueError(v)
        return (int(v[0]), int(v[1]))
    elif isinstance(v, int):
        return (v, v)
    raise ValueError(v)
# %%
def conv2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> t.Tensor:
    '''Like torch's conv2d using bias=False

    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)


    Returns: shape (batch, out_channels, output_height, output_width)
    '''
    stride = force_pair(stride)
    padding = force_pair(padding)
    out_channels, in_channels, kernel_height, kernel_width = weights.shape
    batch, in_channels, height, width = x.shape
    x_padded = pad2d(x, padding[1], padding[1], padding[0], padding[0], 0.0)
    new_height = int(np.floor((height + 2 * padding[0] - kernel_height) / stride[0])) + 1
    new_width = int(np.floor((width + 2 * padding[1] - kernel_width) / stride[1])) + 1
    x_exp = x_padded.as_strided(
        size=(batch, in_channels, kernel_height, new_height, kernel_width, new_width),
        stride=(x_padded.stride()[0], x_padded.stride()[1], x_padded.stride()[2], x_padded.stride()[2] * stride[0], x_padded.stride()[3], x_padded.stride()[3] * stride[1])
    )

    return einsum('b in_ch kernel_h h kernel_w w, o_ch in_ch kernel_h kernel_w -> b o_ch h w', x_exp, weights)

utils_w0d2.test_conv2d(conv2d)
# %%
def maxpool2d(x: t.Tensor, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 0
) -> t.Tensor:
    '''Like PyTorch's maxpool2d.

    x: shape (batch, channels, height, width)
    stride: if None, should be equal to the kernel size

    Return: (batch, channels, out_height, output_width)
    '''
    kernel_height, kernel_width = force_pair(kernel_size)
    stride = force_pair(stride) if stride is not None else force_pair(kernel_size)
    padding = force_pair(padding)
    batch, in_channels, height, width = x.shape
    x_padded = pad2d(x, padding[1], padding[1], padding[0], padding[0], -np.inf)
    new_height = int(np.floor((height + 2 * padding[0] - kernel_height) / stride[0])) + 1
    new_width = int(np.floor((width + 2 * padding[1] - kernel_width) / stride[1])) + 1
    x_exp = x_padded.as_strided(
        size=(batch, in_channels, kernel_height, new_height, kernel_width, new_width),
        stride=(x_padded.stride()[0], x_padded.stride()[1], x_padded.stride()[2], x_padded.stride()[2] * stride[0], x_padded.stride()[3], x_padded.stride()[3] * stride[1])
    )
    return t.amax(x_exp, dim=(2, 4))

utils_w0d2.test_maxpool2d(maxpool2d)
# %%
class MaxPool2d(nn.Module):
    def __init__(self, kernel_size: IntOrPair, stride: Optional[IntOrPair] = None, padding: IntOrPair = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Call the functional version of maxpool2d.'''
        return maxpool2d(x, self.kernel_size, self.stride, self.padding)

    def extra_repr(self) -> str:
        '''Add additional information to the string representation of this class.'''
        return f"kernel_size {self.kernel_size}, stride {self.stride}, padding {self.padding}"

utils_w0d2.test_maxpool2d_module(MaxPool2d)
m = MaxPool2d(kernel_size=3, stride=2, padding=1)
# %%
class ReLU(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        return t.max(x.new_full(x.shape, 0.0), x)

utils_w0d2.test_relu(ReLU)
# %%
class Flatten(nn.Module):
    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, input: t.Tensor) -> t.Tensor:
        '''Flatten out dimensions from start_dim to end_dim, inclusive of both.
        '''
        actual_end_dim = len(input) + 1 if self.end_dim == -1 else self.end_dim + 1 # I am confused
        left_dims = input.shape[:self.start_dim]
        flattened_dims = input.shape[self.start_dim:actual_end_dim]
        right_dims = input.shape[actual_end_dim:]
        flattened_shape = list(left_dims) + [np.prod(flattened_dims)] + list(right_dims)
        return t.reshape(input, flattened_shape)

    def extra_repr(self) -> str:
        return f"start_dim {self.start_dim}, end_dim {self.end_dim}"

utils_w0d2.test_flatten(Flatten)
# %%
class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias=True):
        '''A simple linear (technically, affine) transformation.

        The fields should be named `weight` and `bias` for compatibility with PyTorch.
        If `bias` is False, set `self.bias` to None.
        '''
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter((t.rand((self.out_features, self.in_features)) * 2 - 1) / np.sqrt(self.in_features))
        self.bias = nn.Parameter((t.rand((self.out_features)) * 2 - 1) / np.sqrt(self.in_features)) if bias else None

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (*, in_features)
        Return: shape (*, out_features)
        '''
        return einsum('o_f i_f, batch i_f -> batch o_f', self.weight, x) + (self.bias if self.bias is not None else 0)

    def extra_repr(self) -> str:
        return f"weight {self.weight}, bias {self.bias}"

utils_w0d2.test_linear_forward(Linear)
utils_w0d2.test_linear_parameters(Linear)
utils_w0d2.test_linear_no_bias(Linear)
# %%
class Conv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: IntOrPair, stride: IntOrPair = 1, padding: IntOrPair = 0):
        '''
        Same as torch.nn.Conv2d with bias=False.

        Name your weight field `self.weight` for compatibility with the PyTorch version.
        '''
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = stride
        self.padding = padding

        scaling_factor = 1 / np.sqrt(self.in_channels * self.kernel_size[0] * self.kernel_size[1])
        self.weight = nn.Parameter((t.rand((self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])) * 2 - 1) * scaling_factor)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Apply the functional conv2d you wrote earlier.'''
        return conv2d(x, self.weight, self.stride, self.padding)

    def extra_repr(self) -> str:
        return f"weight {self.weight}"

utils_w0d2.test_conv2d_module(Conv2d)
# %%
class BatchNorm2d(nn.Module):
    running_mean: t.Tensor         # shape: (num_features,)
    running_var: t.Tensor          # shape: (num_features,)
    num_batches_tracked: t.Tensor  # shape: ()

    def __init__(self, num_features: int, eps=1e-05, momentum=0.1):
        '''Like nn.BatchNorm2d with track_running_stats=True and affine=True.

        Name the learnable affine parameters `weight` and `bias` in that order.
        '''
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(
            t.ones(num_features)
        )
        self.bias = nn.Parameter(
            t.zeros(num_features)
        )
        self.register_buffer('running_mean', t.zeros(num_features))
        self.register_buffer('running_var', t.ones(num_features))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('num_batches_tracked', t.tensor(0))


    def forward(self, x: t.Tensor) -> t.Tensor:
        '''Normalize each channel.

        Compute the variance using `torch.var(x, unbiased=False)`
        Hint: you may also find it helpful to use the argument `keepdim`.

        x: shape (batch, channels, height, width)
        Return: shape (batch, channels, height, width)
        '''
        batches, colours, height, width = x.shape
        x_mean = x.mean(dim=[0, 2, 3])
        x_var = x.var(dim=[0, 2, 3])
        expander = lambda v: repeat(v, 'c -> b c h w', b=batches, h=height, w=width)
        if self.training:
            self.running_mean = self.momentum * x_mean + (1 - self.momentum) * self.running_mean if self.training else self.running_mean
            self.running_var = self.momentum * x_var + (1 - self.momentum) * self.running_var if self.training else self.running_mean
            x_mean = expander(x_mean)
            x_var = expander(x_var)
            self.num_batches_tracked += 1
        else:
            x_mean = expander(self.running_mean)
            x_var = expander(self.running_var)
        w = expander(self.weight)
        b = expander(self.bias)
        x_scaled = (x - x_mean) / t.sqrt(x_var + self.eps)
        y = x_scaled * w + b
        return y

    def extra_repr(self) -> str:
        return f'weight={self.weight} bias={self.bias} mu={self.running_mean} var={self.running_var}'

utils_w0d3.test_batchnorm2d_module(BatchNorm2d)
utils_w0d3.test_batchnorm2d_forward(BatchNorm2d)
utils_w0d3.test_batchnorm2d_running_mean(BatchNorm2d)

# %%
class AveragePool(nn.Module):
    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, channels, height, width)
        Return: shape (batch, channels)
        '''
        return x.mean(dim=[2, 3])
# %%