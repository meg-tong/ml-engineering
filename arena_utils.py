import warnings
from typing import Callable, List, Optional

import ipywidgets as wg
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.graph_objs as go
import torch
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchvision
import transformers
from einops import rearrange, repeat
from IPython.display import display
from plotly.subplots import make_subplots
from sklearn.datasets import make_moons
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import random

Arr = np.ndarray

def test_DFT_func(DFT_1d, x=np.linspace(-1, 1), function=lambda x: x**2) -> None:
    
    y = function(x)
    y_DFT = DFT_1d(y)
    y_reconstructed = DFT_1d(y_DFT, inverse=True)
    
    np.testing.assert_allclose(y, y_reconstructed, atol=1e-10)
    
def test_integrate_function(integrate_function):
    
    func = lambda x: np.sin(x) ** 2
    x0 = -np.pi
    x1 = np.pi
    
    integral_approx = integrate_function(func, x0, x1)
    integral_true = 0.5 * (x1 - x0)
    
    np.testing.assert_allclose(integral_true, integral_approx, atol=1e-10)

def test_integrate_product(integrate_product):
    
    func1 = np.sin
    func2 = np.cos 
    x0 = -np.pi
    x1 = np.pi
    
    integrate_product_approx = integrate_product(func1, func2, x0, x1)
    integrate_product_true = 0
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)
    
    integrate_product_approx = integrate_product(func1, func1, x0, x1)
    integrate_product_true = 0.5 * (x1 - x0)
    np.testing.assert_allclose(integrate_product_true, integrate_product_approx, atol=1e-10)

def create_interactive_fourier_graph(calculate_fourier_series: Callable, func: Callable):

    label = wg.Label("Number of terms in Fourier series: ")

    slider = wg.IntSlider(min=0, max=50, value=0)

    x = np.linspace(-np.pi, np.pi, 1000)
    y = func(x)

    fig = go.FigureWidget(
        data = [
            go.Scatter(x=x, y=y, name="Original function", mode="lines"),
            go.Scatter(x=x, y=y, name="Reconstructed function", mode="lines")
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100)
    )

    def respond_to_slider(change):
        max_freq = slider.value
        coeffs, func_approx = calculate_fourier_series(func, max_freq)
        fig.data[1].y = np.vectorize(func_approx)(x)

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])

TARGET_FUNC = np.sin
NUM_FREQUENCIES = 4
TOTAL_STEPS = 4000
LEARNING_RATE = 1e-6
    
def get_title_from_coeffs(a_0, A_n, B_n):
    A_n_coeffs = " + ".join([f"{a_n:.2f}" + r"\cos{" + (str(n) if n>1 else "") + " x}" for (n, a_n) in enumerate(A_n, 1)])
    B_n_coeffs = " + ".join([f"{b_n:.2f}" + r"\sin{" + (str(n) if n>1 else "") + " x}" for (n, b_n) in enumerate(B_n, 1)])
    return f"{a_0:.2f}" + " + " + A_n_coeffs + " + " + B_n_coeffs + r"$"

def visualise_fourier_coeff_convergence(x, y, y_pred_list, coeffs_list):

    label = wg.Label("Number of steps: ")

    slider = wg.IntSlider(min=0, max=TOTAL_STEPS-100, value=0, step=100)

    fig = go.FigureWidget(
        data = [go.Scatter(x=x, y=y, mode="lines", marker_color="blue")] + [
            go.Scatter(x=x, y=y_pred_list[i], mode="lines", marker_color="rgba(100, 100, 100, 0.1)")
            for i in range(len(y_pred_list))
        ],
        layout = go.Layout(title_text=r"Original vs reconstructed", template="simple_white", margin_t=100, showlegend=False)
    )

    def respond_to_slider(change):
        idx = slider.value // 100
        with fig.batch_update():
            fig.update_layout(title_text = get_title_from_coeffs(*coeffs_list[idx]))
            for i in range(len(list(fig.data))-1):
                fig.data[i+1]["marker"]["color"] = "red" if i == idx else "rgba(100, 100, 100, 0.1)"

    slider.observe(respond_to_slider)

    respond_to_slider("unimportant text to trigger first response")

    box_layout = wg.Layout(border="solid 1px black", padding="20px", margin="20px", width="80%")

    return wg.VBox([wg.HBox([label, slider], layout=box_layout), fig])

def display_array_as_img(img_array):
    """
    Displays a numpy array as an image
    
    Two options:
        img_array.shape = (height, width) -> interpreted as monochrome
        img_array.shape = (3, height, width) -> interpreted as RGB
    """
    shape = img_array.shape
    assert len(shape) == 2 or (shape[0] == 3 and len(shape) == 3), "Incorrect format (see docstring)"
    
    if len(shape) == 3:
        img_array = rearrange(img_array, "c h w -> h w c")
    height, width = img_array.shape[:2]
    
    fig = px.imshow(img_array, zmin=0, zmax=255, color_continuous_scale='gray')
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    fig.update_layout(coloraxis_showscale=False, margin=dict.fromkeys("tblr", 0), height=height, width=width)
    fig.show(config=dict(displayModeBar=False))

def test_einsum_trace(einsum_trace):
    mat = np.random.randn(3, 3)
    np.testing.assert_almost_equal(einsum_trace(mat), np.trace(mat))
    print("All tests in `test_einsum_trace` passed!")

def test_einsum_mv(einsum_mv):
    mat = np.random.randn(2, 3)
    vec = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_mv(mat, vec), mat @ vec)
    print("All tests in `test_einsum_mv` passed!")

def test_einsum_mm(einsum_mm):
    mat1 = np.random.randn(2, 3)
    mat2 = np.random.randn(3, 4)
    np.testing.assert_almost_equal(einsum_mm(mat1, mat2), mat1 @ mat2)
    print("All tests in `test_einsum_mm` passed!")

def test_einsum_inner(einsum_inner):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(3)
    np.testing.assert_almost_equal(einsum_inner(vec1, vec2), np.dot(vec1, vec2))
    print("All tests in `test_einsum_inner` passed!")

def test_einsum_outer(einsum_outer):
    vec1 = np.random.randn(3)
    vec2 = np.random.randn(4)
    np.testing.assert_almost_equal(einsum_outer(vec1, vec2), np.outer(vec1, vec2))
    print("All tests in `test_einsum_outer` passed!")


def test_trace(trace_fn):
    for n in range(10):
        assert trace_fn(t.zeros((n, n), dtype=t.long)) == 0, f"Test failed on zero matrix with size ({n}, {n})"
        assert trace_fn(t.eye(n, dtype=t.long)) == n, f"Test failed on identity matrix with size ({n}, {n})"
        x = t.randint(0, 10, (n, n))
        expected = t.trace(x)
        actual = trace_fn(x)
        assert actual == expected, f"Test failed on randmly initialised matrix with size ({n}, {n})"
    print("All tests in `test_trace` passed!")

def test_mv(mv_fn):
    mat = t.randn(3, 4)
    vec = t.randn(4)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
    print("All tests in `test_mv` passed!")
    
def test_mv2(mv_fn):
    big = t.randn(30)
    mat = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    vec = big.as_strided(size=(4,), stride=(3,), storage_offset=8)
    mv_actual = mv_fn(mat, vec)
    mv_expected = mat @ vec
    t.testing.assert_close(mv_actual, mv_expected)
    print("All tests in `test_mv2` passed!")
        
def test_mm(mm_fn):
    matA = t.randn(3, 4)
    matB = t.randn(4, 5)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)
    print("All tests in `test_mm` passed!")

def test_mm2(mm_fn):
    big = t.randn(30)
    matA = big.as_strided(size=(3, 4), stride=(2, 4), storage_offset=8)
    matB = big.as_strided(size=(4, 5), stride=(3, 2), storage_offset=8)
    mm_actual = mm_fn(matA, matB)
    mm_expected = matA @ matB
    t.testing.assert_close(mm_actual, mm_expected)
    print("All tests in `test_mm2` passed!")
    
def test_conv1d_minimal(conv1d_minimal, n_tests=20):
    import numpy as np
    for _ in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 30)#width
        ci = np.random.randint(1, 5)
        co = np.random.randint(1, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d_minimal(x, weights)
        torch_output = t.conv1d(x, weights, stride=1, padding=0)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d_minimal` passed!")

def test_conv2d_minimal(conv2d_minimal, n_tests=4):
    """
    Compare against torch.conv2d.
    Due to floating point rounding, they can be quite different in float32 but should be nearly identical in float64.
    """
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d_minimal(x, weights)
        torch_output = t.conv2d(x, weights)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_minimal` passed!")

def test_conv1d(conv1d, n_tests=10):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = np.random.randint(1, 5)
        padding = np.random.randint(0, 5)
        kernel_size = np.random.randint(1, 10)
        x = t.randn((b, ci, h))
        weights = t.randn((co, ci, kernel_size))
        my_output = conv1d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv1d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv1d` passed!")

def test_pad1d(pad1d):
    """Should work with one channel of width 4."""
    x = t.arange(4).float().view((1, 1, 4))
    actual = pad1d(x, 1, 3, -2.0)
    expected = t.tensor([[[-2.0, 0.0, 1.0, 2.0, 3.0, -2.0, -2.0, -2.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d` passed!")


def test_pad1d_multi_channel(pad1d):
    """Should work with two channels of width 2."""
    x = t.arange(4).float().view((1, 2, 2))
    actual = pad1d(x, 0, 2, -3.0)
    expected = t.tensor([[[0.0, 1.0, -3.0, -3.0], [2.0, 3.0, -3.0, -3.0]]])
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad1d_multi_channel` passed!")

def test_pad2d(pad2d):
    """Should work with one channel of 2x2."""
    x = t.arange(4).float().view((1, 1, 2, 2))
    expected = t.tensor([[[
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 3.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]]])
    actual = pad2d(x, 0, 1, 2, 3, 0.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d` passed!")

def test_pad2d_multi_channel(pad2d):
    """Should work with two channels of 2x1."""
    x = t.arange(4).float().view((1, 2, 2, 1))
    expected = t.tensor([[[[-1.0, 0.0], [-1.0, 1.0], [-1.0, -1.0]], [[-1.0, 2.0], [-1.0, 3.0], [-1.0, -1.0]]]])
    actual = pad2d(x, 1, 0, 0, 1, -1.0)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_pad2d_multi_channel` passed!")

def test_conv2d(conv2d, n_tests=5):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w), dtype=t.float64)
        weights = t.randn((co, ci, *kernel_size), dtype=t.float64)
        my_output = conv2d(x, weights, stride=stride, padding=padding)
        torch_output = t.conv2d(x, weights, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d` passed!")

def test_maxpool2d(my_maxpool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = my_maxpool2d(
            x,
            kernel_size,
            stride=stride,
            padding=padding,
        )
        torch_output = t.max_pool2d(
            x,
            kernel_size,
            stride=stride,  # type: ignore (None actually is allowed)
            padding=padding,
        )
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d` passed!")

def test_maxpool2d_module(MaxPool2d, n_tests=20):
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 50)
        w = np.random.randint(10, 50)
        ci = np.random.randint(1, 20)
        stride = None if np.random.random() < 0.5 else tuple(np.random.randint(1, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        kH, kW = kernel_size
        padding = np.random.randint(0, 1 + kH // 2), np.random.randint(0, 1 + kW // 2)
        x = t.randn((b, ci, h, w))
        my_output = MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)

        torch_output = nn.MaxPool2d(
            kernel_size,
            stride=stride,
            padding=padding,
        )(x)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_maxpool2d_module` passed!")

def test_conv2d_module(Conv2d, n_tests=5):
    """
    Your weight should be called 'weight' and have an appropriate number of elements.
    """
    m = Conv2d(4, 5, (3, 3))
    assert isinstance(m.weight, t.nn.parameter.Parameter), "Weight should be registered a parameter!"
    assert m.weight.nelement() == 4 * 5 * 3 * 3
    import numpy as np
    for i in range(n_tests):
        b = np.random.randint(1, 10)
        h = np.random.randint(10, 300)
        w = np.random.randint(10, 300)
        ci = np.random.randint(1, 20)
        co = np.random.randint(1, 20)
        stride = tuple(np.random.randint(1, 5, size=(2,)))
        padding = tuple(np.random.randint(0, 5, size=(2,)))
        kernel_size = tuple(np.random.randint(1, 10, size=(2,)))
        x = t.randn((b, ci, h, w))
        my_conv = Conv2d(in_channels=ci, out_channels=co, kernel_size=kernel_size, stride=stride, padding=padding)
        my_output = my_conv(x)
        torch_output = t.conv2d(x, my_conv.weight, stride=stride, padding=padding)
        t.testing.assert_close(my_output, torch_output)
    print("All tests in `test_conv2d_module` passed!")

def test_relu(ReLU):
    x = t.randn(10) - 0.5
    actual = ReLU()(x)
    expected = F.relu(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_relu` passed!")

def test_flatten(Flatten):
    x = t.arange(24).reshape((2, 3, 4))
    assert Flatten(start_dim=0)(x).shape == (24,)
    assert Flatten(start_dim=1)(x).shape == (2, 12)
    assert Flatten(start_dim=0, end_dim=1)(x).shape == (6, 4)
    assert Flatten(start_dim=0, end_dim=-2)(x).shape == (6, 4)
    print("All tests in `test_flatten` passed!")

def test_linear_forward(Linear):
    """Your Linear should produce identical results to torch.nn given identical parameters."""
    x = t.rand((10, 512))
    yours = Linear(512, 64)
    assert yours.weight.shape == (64, 512), f"Linear layer weights have wrong shape: {yours.weight.shape}, expected shape = (64, 512)"
    assert yours.bias.shape == (64,), f"Linear layer bias has wrong shape: {yours.bias.shape}, expected shape = (64,)"
    official = t.nn.Linear(512, 64)
    yours.weight = official.weight
    yours.bias = official.bias
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_forward` passed!")

def test_linear_parameters(Linear):
    m = Linear(2, 3)
    params = dict(m.named_parameters())
    assert len(params) == 2, f"Your model has {len(params)} recognized Parameters"
    assert list(params.keys()) == [
        "weight",
        "bias",
    ], f"For compatibility with PyTorch, your fields should be named weight and bias, not {tuple(params.keys())}"
    print("All tests in `test_linear_parameters` passed!")

def test_linear_no_bias(Linear):
    
    x = t.rand((10, 512))
    yours = Linear(512, 64, bias=False)

    assert yours.bias is None, "Bias should be None when not enabled."
    assert len(list(yours.parameters())) == 1

    official = nn.Linear(512, 64, bias=False)
    yours.weight = official.weight
    actual = yours(x)
    expected = official(x)
    t.testing.assert_close(actual, expected)
    print("All tests in `test_linear_no_bias` passed!")

def plot_loss_and_accuracy(loss_list, accuracy_list):

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Scatter(y=loss_list, name="Loss"), secondary_y=False)

    n_batches = len(loss_list) // len(accuracy_list)
    x = list(range(n_batches-1, len(loss_list), n_batches))
    fig.add_trace(go.Scatter(y=accuracy_list, x=x, name="Accuracy"), secondary_y=True)

    fig.update_layout(
        title_text="CNN training loss & test accuracy",
        template="simple_white", 
        xaxis_range=[0, len(loss_list)], 
        yaxis2_range=[0, 1],
        yaxis2_tickformat=".0%", 
        hovermode="x unified"
    )

    fig.show()

def test_batchnorm2d_module(BatchNorm2d):
    """The public API of the module should be the same as the real PyTorch version."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.num_features == num_features
    assert isinstance(bn.weight, t.nn.parameter.Parameter), f"weight has wrong type: {type(bn.weight)}"
    assert isinstance(bn.bias, t.nn.parameter.Parameter), f"bias has wrong type: {type(bn.bias)}"
    assert isinstance(bn.running_mean, t.Tensor), f"running_mean has wrong type: {type(bn.running_mean)}"
    assert isinstance(bn.running_var, t.Tensor), f"running_var has wrong type: {type(bn.running_var)}"
    assert isinstance(bn.num_batches_tracked, t.Tensor), f"num_batches_tracked has wrong type: {type(bn.num_batches_tracked)}"
    print("All tests in `test_batchnorm2d_module` passed!")

def test_batchnorm2d_forward(BatchNorm2d):
    """For each channel, mean should be very close to 0 and std kinda close to 1 (because of eps)."""
    num_features = 2
    bn = BatchNorm2d(num_features)
    assert bn.training
    x = t.randn((100, num_features, 3, 4))
    out = bn(x)
    assert x.shape == out.shape
    t.testing.assert_close(out.mean(dim=(0, 2, 3)), t.zeros(num_features))
    t.testing.assert_close(out.std(dim=(0, 2, 3)), t.ones(num_features), atol=1e-3, rtol=1e-3)
    print("All tests in `test_batchnorm2d_forward` passed!")

def test_batchnorm2d_running_mean(BatchNorm2d):
    """Over repeated forward calls with the same data in train mode, the running mean should converge to the actual mean."""
    bn = BatchNorm2d(3, momentum=0.6)
    assert bn.training
    x = t.arange(12).float().view((2, 3, 2, 1))
    mean = t.tensor([3.5000, 5.5000, 7.5000])
    num_batches = 30
    for i in range(num_batches):
        bn(x)
        expected_mean = (1 - (((1 - bn.momentum) ** (i + 1)))) * mean
        t.testing.assert_close(bn.running_mean, expected_mean)
    assert bn.num_batches_tracked.item() == num_batches

    # Large enough momentum and num_batches -> running_mean should be very close to actual mean
    bn.eval()
    actual_eval_mean = bn(x).mean((0, 2, 3))
    t.testing.assert_close(actual_eval_mean, t.zeros(3))
    print("All tests in `test_batchnorm2d_running_mean` passed!")


def compare_my_resnet_to_pytorch(myresnet):
    
    their_state = torchvision.models.resnet34().state_dict().items()
    your_state = myresnet.state_dict().items()
    
    df = pd.DataFrame.from_records(
        [(tk, tuple(tv.shape), mk, tuple(mv.shape)) for ((tk, tv), (mk, mv)) in zip(their_state, your_state)],
        columns=["their name", "their shape", "your name", "your shape"],
    )
    with pd.option_context("display.max_rows", None):  # type: ignore
        display(df)

def print_param_count(*models, display_df=True, use_state_dict=True):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        print(f"Model {i}, total params = {sum([param.numel() for name, param in iterator])}")
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

def plot_results(loss_list, accuracy_list):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(y=loss_list, name="Training loss"))
    fig.update_xaxes(title_text="Num batches observed")
    fig.update_yaxes(title_text="Training loss", secondary_y=False)
    # This next bit of code plots vertical lines corresponding to the epochs
    if len(accuracy_list) > 1:
        for idx, epoch_start in enumerate(np.linspace(0, len(loss_list), len(accuracy_list), endpoint=False)):
            fig.add_vline(x=epoch_start, line_width=3, line_dash="dash", annotation_text=f"Epoch {idx}", annotation_position="top right")
        fig.add_trace(
            go.Scatter(y=accuracy_list, x=np.linspace(0, len(loss_list), len(accuracy_list)), mode="lines", name="Accuracy"),
            secondary_y=True
        )
    fig.update_layout(template="simple_white", title_text="Training loss & accuracy on CIFAR10")
    fig.show()

def show_cifar_images(trainset, rows=3, cols=5):
    
    img = trainset.data[:rows*cols]
    fig = px.imshow(img, facet_col=0, facet_col_wrap=cols)
    for i, j in enumerate(np.arange(rows*cols).reshape(rows, cols)[::-1].flatten()):
            fig.layout.annotations[i].text = trainset.classes[trainset.targets[j]]
    fig.show()


def visualize(dataloader):
    (sample, sample_labels) = next(iter(dataloader))
        
    fig = make_subplots(
        rows=2, cols=5,
        horizontal_spacing=0.02, vertical_spacing=0.02,
        subplot_titles=[str(label.item()) for label in sample_labels[:10]]
    )
    for row in range(2):
        for col in range(5):
            z = repeat((255 * (0.28 + 0.35*sample[5*row+col, 0])).numpy().astype(int), "h w -> h w 3")
            fig.add_trace(go.Image(z=z), row=row+1, col=col+1)
            
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    fig.update_layout(margin=dict(t=50, b=0, r=20, l=20))

    fig.show(config={'displayModeBar': False})

def get_mnist(subsample: Optional[int] = None):
    """Return MNIST data using the provided Tensor class."""
    mnist_train = datasets.MNIST("../data", train=True, download=True)
    mnist_test = datasets.MNIST("../data", train=False)
    if subsample is None:
        subsample = 1
    print("Preprocessing data...")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.28,), (0.35,))])
    train_indexes = range(0, len(mnist_train), subsample)
    train_reduced = [mnist_train[i] for i in train_indexes]
    train_tensors = torch.utils.data.TensorDataset(
        torch.stack([transform(img) for img, label in tqdm(train_reduced, desc="Training data")]),
        torch.tensor([label for img, label in train_reduced]),
    )

    test_indexes = range(0, len(mnist_test), subsample)
    test_reduced = [mnist_test[i] for i in test_indexes]
    test_tensors = torch.utils.data.TensorDataset(
        torch.stack([transform(img) for img, label in tqdm(test_reduced, desc="Test data")]),
        torch.tensor([label for img, label in test_reduced]),
    )

    train_loader = torch.utils.data.DataLoader(train_tensors, shuffle=True, batch_size=512)
    test_loader = torch.utils.data.DataLoader(test_tensors, batch_size=512)
    return train_loader, test_loader


def test_log_back(log_back):
    a = np.array([1, np.e, np.e**np.e])
    b = np.log(a)
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = log_back(grad_out, b, a)
    expected = [2.0, 2.0 / np.e, 2.0 / (np.e**np.e)]
    np.testing.assert_allclose(actual, expected)
    print("All tests in `test_einsum_inner` passed!")

def test_unbroadcast(unbroadcast):
    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 20.0).all(), "Each element in the small array appeared 20 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (5, 1, 2, 1, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 5.0).all(), "Each element in the small array appeared 5 times in the large array."

    small = np.ones((2, 1, 3))
    large = np.broadcast_to(small, (2, 4, 3))
    out = unbroadcast(large, small)
    assert out.shape == small.shape
    assert (out == 4.0).all(), "Each element in the small array appeared 4 times in the large array."
    print("All tests in `test_unbroadcast` passed!")

def test_multiply_back(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = np.array([2])
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = np.array([4.0, 4.0, 4.0])
    assert actual.shape == expected.shape, f"Output has wrong shape: {actual.shape}, expected {expected.shape}"
    assert np.allclose(actual, expected), "Correct shape, but wrong values"
    actual = multiply_back1(grad_out, c, a, b)
    expected = np.array([12.0])
    assert actual.shape == expected.shape, f"Output has wrong shape: {actual.shape}, expected {expected.shape}"
    assert np.allclose(actual, expected), "Correct shape, but wrong values"
    print("All tests in `test_multiply_back` passed!")

def test_multiply_back_float(multiply_back0, multiply_back1):
    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back0(grad_out, c, a, b)
    expected = [4.0, 4.0, 4.0]
    np.testing.assert_allclose(actual, expected)
    a = np.array([1, 2, 3])
    b = 2
    c = a * b
    grad_out = np.array([2.0, 2.0, 2.0])
    actual = multiply_back1(grad_out, c, b, a)
    expected = [4.0, 4.0, 4.0]
    np.testing.assert_allclose(actual, expected)
    print("All tests in `test_multiply_back_float` passed!")

def test_forward_and_back(forward_and_back):
    a = np.array([1, 2, 3])
    b = np.array([2, 3, 1])
    c = np.array([10])
    dg_da, dg_db, dg_dc = forward_and_back(a, b, c)
    expected_dg_da = np.array([1, 1 / 2, 1 / 3])
    expected_dg_db = np.array([1 / 2, 1 / 3, 1])
    expected_dg_dc = np.array([0.13028834])
    np.testing.assert_allclose(dg_da, expected_dg_da)
    np.testing.assert_allclose(dg_db, expected_dg_db)
    np.testing.assert_allclose(dg_dc, expected_dg_dc)
    print("All tests in `test_forward_and_back` passed!")

def test_back_func_lookup(BackwardFuncLookup):
    backward_funcs = BackwardFuncLookup()
    backward_funcs.add_back_func(np.log, 0, np.exp)
    assert backward_funcs.get_back_func(np.log, 0) == np.exp
    backward_funcs.add_back_func(np.multiply, 0, np.divide)
    assert backward_funcs.get_back_func(np.multiply, 0) == np.divide
    backward_funcs.add_back_func(np.multiply, 1, np.add)
    assert backward_funcs.get_back_func(np.multiply, 1) == np.add
    print("All tests in `test_back_func_lookup` passed!")

def test_log(Tensor, log_forward):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = log_forward(a)
    np.testing.assert_allclose(b.array, [1, np.e])
    assert b.requires_grad == True, "Should require grad because input required grad."
    assert b.is_leaf == False
    assert b.recipe is not None
    assert len(b.recipe.parents) == 1 and b.recipe.parents[0] is a
    assert len(b.recipe.args) == 1 and b.recipe.args[0] is a.array
    assert b.recipe.kwargs == {}
    assert b.recipe.func is np.log
    c = log_forward(b)
    np.testing.assert_almost_equal(c.array, [0, 1])
    print("All tests in `test_log` passed!")

def test_log_no_grad(Tensor, log_forward):
    d = Tensor([1, np.e])
    e = log_forward(d)
    assert e.requires_grad == False, "Should not require grad because input did not."
    assert e.recipe is None
    np.testing.assert_allclose(e.array, [0, 1])
    print("All tests in `test_log_no_grad` passed!")

def test_multiply(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = multiply(a, b)
    assert c.requires_grad == True, "Should require grad because input required grad."
    assert c.is_leaf == False
    assert c.recipe is not None
    assert len(c.recipe.parents) == 2 and c.recipe.parents[0] is a and c.recipe.parents[1] is b
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply` passed!")

def test_multiply_no_grad(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=False)
    b = Tensor([[0], [1], [10]], requires_grad=False)
    c = multiply(a, b)
    assert c.requires_grad == False, "Should not require grad because input did not require grad."
    assert c.recipe is None
    expected = np.array([[0, 0, 0, 0], [0, 1, 2, 3], [0, 10, 20, 30]])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply_no_grad` passed!")

def test_multiply_float(Tensor, multiply):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(a, b)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[0] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is a.array and c.recipe.args[1] is b
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.testing.assert_allclose(c.array, expected)
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = 3
    c = multiply(b, a)
    assert c.requires_grad == True
    assert c.recipe is not None
    assert len(c.recipe.parents) == 1 and c.recipe.parents[1] is a
    assert len(c.recipe.args) == 2 and c.recipe.args[0] is b and c.recipe.args[1] is a.array
    assert c.recipe.kwargs == {}
    assert c.recipe.func is np.multiply
    expected = np.array([0, 3, 6, 9])
    np.testing.assert_allclose(c.array, expected)
    print("All tests in `test_multiply_float` passed!")

def test_sum(wrap_forward_fn, Tensor):
    # This tests keyword arguments
    def _sum(x: Arr, dim=None, keepdim=False) -> Arr:
        """Like torch.sum, calling np.sum internally."""
        return np.sum(x, axis=dim, keepdims=keepdim)
    global sum
    sum = wrap_forward_fn(_sum)
    a = Tensor(np.array([[0.0, 1.0], [2.0, 3.0]]), requires_grad=True)
    assert a.sum(0).shape == (2,)
    assert a.sum(0, True).shape == (1, 2)
    print("All tests in `test_sum` passed!")
    


class Node:
    def __init__(self, *children):
        self.children = list(children)


def get_children(node: Node) -> List[Node]:
    return node.children

def test_topological_sort_linked_list(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    expected = [z, y, x]
    for e, a in zip(expected, topological_sort(x, get_children)):
        assert e is a
    print("All tests in `test_topological_sort_linked_list` passed!")

def test_topological_sort_branching(topological_sort):
    z = Node()
    y = Node()
    x = Node(y, z)
    w = Node(x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw" or out == "yzxw"
    print("All tests in `test_topological_sort_branching` passed!")

def test_topological_sort_rejoining(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    w = Node(z, x)
    name_lookup = {w: "w", x: "x", y: "y", z: "z"}
    out = "".join([name_lookup[n] for n in topological_sort(w, get_children)])
    assert out == "zyxw"
    print("All tests in `test_topological_sort_rejoining` passed!")

def test_topological_sort_cyclic(topological_sort):
    z = Node()
    y = Node(z)
    x = Node(y)
    z.children = [x]
    try:
        topological_sort(x, get_children)
    except:
        assert True
    else:
        assert False
    print("All tests in `test_topological_sort_cyclic` passed!")

def test_backprop(Tensor):
    a = Tensor([np.e, np.e**np.e], requires_grad=True)
    b = a.log()
    c = b.log()
    c.backward(end_grad=np.array([1.0, 1.0]))
    assert c.grad is None
    assert b.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, 1 / b.array / a.array)
    print("All tests in `test_backprop` passed!")

def test_backprop_branching(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=True)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert np.allclose(b.grad.array, a.array)
    print("All tests in `test_backprop_branching` passed!")

def test_backprop_requires_grad_false(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1, 2, 3], requires_grad=False)
    c = a * b
    c.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert np.allclose(a.grad.array, b.array)
    assert b.grad is None
    print("All tests in `test_backprop_requires_grad_false` passed!")

def test_backprop_float_arg(Tensor):
    a = Tensor([1, 2, 3], requires_grad=True)
    b = 2
    c = a * b
    d = 2
    e = d * c
    e.backward(end_grad=np.array([1.0, 1.0, 1.0]))
    assert e.grad is None
    assert c.grad is None
    assert a.grad is not None
    assert np.allclose(a.grad.array, np.array([4.0, 4.0, 4.0]))
    print("All tests in `test_backprop_float_arg` passed!")

def test_negative_back(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    b = -a
    c = -b
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, [[1.0, 1.0, 1.0]])
    print("All tests in `test_negative_back` passed!")

def test_exp_back(Tensor):
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    b.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, 1 / np.e, 0, np.e)
    a = Tensor([-1.0, 0.0, 1.0], requires_grad=True)
    b = a.exp()
    c = b.exp()
    c.backward(end_grad=np.array([[1.0, 1.0, 1.0]]))
    def d(x):
        return (np.e**x) * (np.e ** (np.e**x))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, *[d(x) for x in a.array])
    print("All tests in `test_exp_back` passed!")

def test_reshape_back(Tensor):
    a = Tensor([1, 2, 3, 4, 5, 6], requires_grad=True)
    b = a.reshape((3, 2))
    b.backward(end_grad=np.array([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]))
    assert a.grad is not None and np.allclose(a.grad.array, np.ones(6))
    print("All tests in `test_reshape_back` passed!")

def test_permute_back(Tensor):
    a = Tensor(np.arange(24).reshape((2, 3, 4)), requires_grad=True)
    out = a.permute((2, 0, 1))
    out.backward(np.arange(24).reshape((4, 2, 3)))
    assert a.grad is not None
    np.testing.assert_allclose(
        a.grad.array,
        np.array([
            [[0.0, 6.0, 12.0, 18.0], [1.0, 7.0, 13.0, 19.0], [2.0, 8.0, 14.0, 20.0]],
            [[3.0, 9.0, 15.0, 21.0], [4.0, 10.0, 16.0, 22.0], [5.0, 11.0, 17.0, 23.0]],
        ]),
    )
    print("All tests in `test_permute_back` passed!")

def test_expand(Tensor):
    a = Tensor(np.ones((2, 1, 3)), requires_grad=True)
    b = a.expand((5, 1, 2, 4, 3))
    b.backward(np.full_like(b.array, 10.0))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 20 * 10.0).all()
    print("All tests in `test_expand` passed!")

def test_expand_negative_length(Tensor):
    a = Tensor([1.0, 2.0, 3.0, 4.0, 5.0], requires_grad=True)
    b = a.expand((3, 2, -1))
    assert b.shape == (3, 2, 5)
    b.backward(end_grad=np.ones(b.shape))
    assert a.grad is not None and a.grad.shape == a.array.shape
    assert (a.grad.array == 6).all()
    print("All tests in `test_expand_negative_length` passed!")

def test_sum_keepdim_false(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(0)
    c = b.sum(0)
    c.backward(np.array(2))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 2).all()
    print("All tests in `test_sum_keepdim_false` passed!")

def test_sum_keepdim_true(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum(1, keepdim=True)
    c = a.sum(0, keepdim=True)
    np.testing.assert_almost_equal(c.array, np.array([[5.0, 7.0, 9.0, 11.0, 13.0]]))
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 1).all()
    print("All tests in `test_sum_keepdim_true` passed!")

def test_sum_dim_none(Tensor):
    a = Tensor(np.array([[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0]]), requires_grad=True)
    b = a.sum()
    b.backward(np.array(4))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 4).all()
    print("All tests in `test_sum_dim_none` passed!")

def test_getitem_int(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[1]
    c = b.sum(0)
    c.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [10, 10, 10]]))
    print("All tests in `test_getitem_int` passed!")

def test_getitem_tuple(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    b = a[(1, 2)]
    b.backward(np.array(10.0))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([[0, 0, 0], [0, 0, 10]]))
    print("All tests in `test_getitem_tuple` passed!")

def test_getitem_integer_array(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = np.array([0, 1, 0, 1, 0]), np.array([0, 0, 1, 2, 0])
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))
    print("All tests in `test_getitem_integer_array` passed!")

def test_getitem_integer_tensor(Tensor):
    a = Tensor([[0, 1, 2], [3, 4, 5]], requires_grad=True)
    index = Tensor(np.array([0, 1, 0, 1, 0])), Tensor(np.array([0, 0, 1, 2, 0]))
    out = a[index]
    out.sum().backward(np.array(10.0))
    assert a.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[20, 10, 0], [10, 0, 10]]))
    print("All tests in `test_getitem_integer_tensor` passed!")

def test_add_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a + b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == 4).all()
    print("All tests in `test_add_broadcasted` passed!")

def test_subtract_broadcasted(Tensor):
    a = Tensor([0, 1, 2, 3], requires_grad=True)
    b = Tensor([[0], [1], [10]], requires_grad=True)
    c = a - b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == 3).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert (b.grad.array == -4).all()
    print("All tests in `test_subtract_broadcasted` passed!")

def test_truedivide_broadcasted(Tensor):
    a = Tensor([0, 6, 12, 18], requires_grad=True)
    b = Tensor([[1], [2], [3]], requires_grad=True)
    c = a / b
    c.backward(end_grad=np.ones(c.shape))
    assert a.grad is not None
    assert a.grad.shape == a.shape
    assert (a.grad.array == (1 + 1 / 2 + 1 / 3)).all()
    assert b.grad is not None
    assert b.grad.shape == b.shape
    assert np.equal(b.grad.array, np.array([[-36.0], [-9.0], [-4.0]])).all()
    print("All tests in `test_truedivide_broadcasted` passed!")

def test_maximum(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([-1, 1, 3], requires_grad=True)
    out = a.maximum(b)
    np.testing.assert_allclose(out.array, [0, 1, 3])
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.array, [1, 0.5, 0])
    np.testing.assert_allclose(b.grad.array, [0, 0.5, 1])
    print("All tests in `test_maximum` passed!")

def test_maximum_broadcasted(Tensor):
    a = Tensor([0, 1, 2], requires_grad=True)
    b = Tensor([[-1], [1], [3]], requires_grad=True)
    out = a.maximum(b)
    np.testing.assert_allclose(out.array, np.array([[0, 1, 2], [1, 1, 2], [3, 3, 3]]))
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([1.0, 1.5, 2.0]))
    assert b.grad is not None and np.allclose(b.grad.array, np.array([[0.0], [1.5], [3.0]]))
    print("All tests in `test_maximum_broadcasted` passed!")

def test_relu(Tensor):
    a = Tensor([-1, 0, 1], requires_grad=True)
    out = a.relu()
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None and np.allclose(a.grad.array, np.array([0, 0.5, 1.0]))
    print("All tests in `test_relu` passed!")

def test_matmul2d(Tensor):
    a = Tensor(np.arange(-3, 3).reshape((2, 3)), requires_grad=True)
    b = Tensor(np.arange(-4, 5).reshape((3, 3)), requires_grad=True)
    out = a @ b
    out.backward(end_grad=np.ones(out.shape))
    assert a.grad is not None
    assert b.grad is not None
    np.testing.assert_allclose(a.grad.array, np.array([[-9, 0, 9], [-9, 0, 9]]))
    np.testing.assert_allclose(b.grad.array, np.array([[-3, -3, -3], [-1, -1, -1], [1, 1, 1]]))
    print("All tests in `test_matmul2d` passed!")

def test_cross_entropy(Tensor, cross_entropy):
    logits = Tensor([
        [float("-inf"), float("-inf"), float("-inf"), 0], 
        [1/4, 1/4, 1/4, 1/4], 
        [float("-inf"), 0, 0, 0]
    ])
    true_labels = Tensor([3, 0, 0])
    expected = Tensor([0.0, np.log(4), float("inf")])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual = cross_entropy(logits, true_labels)
    np.testing.assert_allclose(actual.array, expected.array)
    print("All tests in `test_cross_entropy` passed!")


def test_embedding(Embedding):
    """Indexing into the embedding should fetch the corresponding rows of the embedding."""
    emb = Embedding(6, 100)
    out = emb(t.tensor([1, 3, 5], dtype=t.int64))
    t.testing.assert_close(out[0], emb.weight[1])
    t.testing.assert_close(out[1], emb.weight[3])
    t.testing.assert_close(out[2], emb.weight[5])

    emb = Embedding(30000, 500)
    t.testing.assert_close(emb.weight.std().item(), 1.0, rtol=0, atol=0.005)

def test_layernorm_mean_1d(LayerNorm):
    """If an integer is passed, this means normalize over the last dimension which should have that size."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10)
    out = ln1(x)
    max_mean = out.mean(-1).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_1d` passed.")

def test_layernorm_mean_2d(LayerNorm):
    """If normalized_shape is 2D, should normalize over both the last two dimensions."""
    x = t.randn(20, 10)
    ln1 = LayerNorm((20, 10))
    out = ln1(x)
    max_mean = out.mean((-1, -2)).abs().max().item()
    assert max_mean < 1e-5, f"Normalized mean should be about 0, got {max_mean}"
    print(f"All tests in `test_layernorm_mean_2d` passed.")

def test_layernorm_std(LayerNorm):
    """If epsilon is small enough and no elementwise_affine, the output variance should be very close to 1."""
    x = t.randn(20, 10)
    ln1 = LayerNorm(10, eps=1e-11, elementwise_affine=False)
    out = ln1(x)
    var_diff = (1 - out.var(-1, unbiased=False)).abs().max().item()
    assert var_diff < 1e-6, f"Var should be about 1, off by {var_diff}"
    print(f"All tests in `test_layernorm_std` passed.")

def test_layernorm_exact(LayerNorm):
    """Your LayerNorm's output should match PyTorch for equal epsilon, up to floating point rounding error.
    This test uses float64 and the result should be extremely tight.
    """
    x = t.randn(2, 3, 4, 5)
    # Use large epsilon to make sure it fails if they forget it
    ln1 = LayerNorm((5,), eps=1e-2)
    ln2 = t.nn.LayerNorm((5,), eps=1e-2)  # type: ignore
    actual = ln1(x)
    expected = ln2(x)
    t.testing.assert_close(actual, expected)
    print(f"All tests in `test_layernorm_exact` passed.")

def test_layernorm_backward(LayerNorm):
    """The backwards pass should also match PyTorch exactly."""
    x = t.randn(10, 3)
    x2 = x.clone()
    x.requires_grad_(True)
    x2.requires_grad_(True)

    # Without parameters, should be deterministic
    ref = nn.LayerNorm(3, elementwise_affine=False)
    ref.requires_grad_(True)
    ref(x).sum().backward()

    ln = LayerNorm(3, elementwise_affine=False)
    ln.requires_grad_(True)
    ln(x2).sum().backward()
    # Use atol since grad entries are supposed to be zero here
    assert isinstance(x.grad, t.Tensor)
    assert isinstance(x2.grad, t.Tensor)
    t.testing.assert_close(x.grad, x2.grad, atol=1e-5, rtol=1e-5)
    print(f"All tests in `test_layernorm_backward` passed.")

def test_dropout_eval(Dropout):
    dropout = Dropout(p=0.1).eval()
    x = t.randn((3, 4))
    t.testing.assert_close(x, dropout(x), msg="Failed on eval mode (shouldn't change tensor).")
    print(f"All tests in `test_dropout_eval` passed.")

def test_dropout_training(Dropout):

    dropout = Dropout(p=0).train()
    x = t.rand((1000, 1000))
    t.testing.assert_close(x, dropout(x), msg="Failed on p=0 (dropout shouldn't change tensor)")

    for p in (0.1, 0.5, 0.9):
        dropout = Dropout(p=p).train()
        x = t.rand((1000, 1000))
        x_dropout = dropout(x)

        close_to_zero = t.abs(x_dropout) < 0.001
        fraction_close_to_zero = close_to_zero.sum() / close_to_zero.numel()
        close_to_ratio = t.abs(x_dropout / x - (1 / (1 - p))) < 0.001
        fraction_close_to_ratio = close_to_ratio.sum() / close_to_ratio.numel()

        assert abs(fraction_close_to_zero - p) < 0.01, f"p={p}, Wrong number of values set to zero"
        assert fraction_close_to_zero + fraction_close_to_ratio > 0.995, f"p={p}, Incorrect scaling"
    
    print(f"All tests in `test_dropout_training` passed.")

def plot_gelu(GELU):
    gelu = GELU()
    x = t.linspace(-5, 5, steps=100)
    px.line(x=x, y=gelu(x), template="ggplot2").show()


def print_param_count(*models, display_df=True, use_state_dict=False):
    """
    display_df: bool
        If true, displays styled dataframe
        if false, returns dataframe

    use_state_dict: bool
        If true, uses model.state_dict() to construct dataframe
            This will include buffers, not just params
        If false, uses model.named_parameters() to construct dataframe
            This misses out buffers (more useful for GPT)
    """
    df_list = []
    gmap_list = []
    for i, model in enumerate(models, start=1):
        print(f"Model {i}, total params = {sum([param.numel() for name, param in model.named_parameters()])}")
        iterator = model.state_dict().items() if use_state_dict else model.named_parameters()
        df = pd.DataFrame([
            {f"name_{i}": name, f"shape_{i}": tuple(param.shape), f"num_params_{i}": param.numel()}
            for name, param in iterator
        ]) if (i == 1) else pd.DataFrame([
            {f"num_params_{i}": param.numel(), f"shape_{i}": tuple(param.shape), f"name_{i}": name}
            for name, param in iterator
        ])
        display(df)
        df_list.append(df)
        gmap_list.append(np.log(df[f"num_params_{i}"]))
    df = df_list[0] if len(df_list) == 1 else pd.concat(df_list, axis=1).fillna(0)
    for i in range(1, len(models) + 1):
        df[f"num_params_{i}"] = df[f"num_params_{i}"].astype(int)
    if len(models) > 1:
        param_counts = [df[f"num_params_{i}"].values.tolist() for i in range(1, len(models) + 1)]
        if all([param_counts[0] == param_counts[i] for i in range(1, len(param_counts))]):
            print("All parameter counts match!")
        else:
            print("Parameter counts don't match up exactly.")
    if display_df:
        s = df.style
        for i in range(1, len(models) + 1):
            s = s.background_gradient(cmap="viridis", subset=[f"num_params_{i}"], gmap=gmap_list[i-1])
        with pd.option_context("display.max_rows", 1000):
            display(s)
    else:
        return df

def test_load_pretrained_weights(model, tokenizer):

    model.eval()
    device = next(model.parameters()).device
    
    def encode(text: str) -> t.Tensor:
        """Return a Tensor of shape (batch=1, seq)."""
        return tokenizer(text, return_tensors="pt")["input_ids"].to(device)

    prompt = "Former President of the United States of America, George"
    input_ids = encode(prompt)
    with t.inference_mode():
        output = model(input_ids)
        logits = output[0, -1] if isinstance(output, t.Tensor) else output.logits[0, -1]
    topk = t.topk(logits, k=10).indices
    next_tokens = tokenizer.batch_decode(topk.reshape(-1, 1))
    print("Prompt: ", prompt)
    print("Your model's top 10 predictions: ", next_tokens)
    assert " Washington" in next_tokens
    assert " Bush" in next_tokens

def test_make_additive_attention_mask(make_additive_attention_mask):
    from solutions_build_bert import \
        make_additive_attention_mask as make_additive_attention_mask_soln
    arr = t.randint(low=0, high=2, size=(3, 4))
    expected = make_additive_attention_mask_soln(arr)
    actual = make_additive_attention_mask(arr)
    t.testing.assert_close(expected, actual)

def _get_moon_data(unsqueeze_y=False):
    X, y = make_moons(n_samples=512, noise=0.05, random_state=354)
    X = t.tensor(X, dtype=t.float32)
    y = t.tensor(y, dtype=t.int64)
    if unsqueeze_y:
        y = y.unsqueeze(-1)
    return DataLoader(TensorDataset(X, y), batch_size=128, shuffle=True)

def _train_with_opt(model, opt):
    dl = _get_moon_data()
    for i, (X, y) in enumerate(dl):
        opt.zero_grad()
        loss = F.cross_entropy(model(X), y)
        loss.backward()
        opt.step()

def _train_with_scheduler(model, opt, scheduler):
    dl = _get_moon_data()
    for epoch in range(20):
        for i, (X, y) in enumerate(dl):
            opt.zero_grad()
            loss = F.cross_entropy(model(X), y)
            loss.backward()
            opt.step()
        scheduler.step()

class Net(nn.Module):
    def __init__(self, in_dim: int=3, hidden_dim: int=5, out_dim: int=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.Linear(2, 5), nn.ReLU())
        self.classifier = nn.Sequential(nn.Linear(5, 3), nn.ReLU())
    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.classifier(self.base(x))

def construct_param_config_from_description(description, model):
    param_config = []
    for param_group in description:
        param_group_ = param_group.copy()
        param_group_["params"] = getattr(model, param_group_["params"]).parameters()
        param_config.append(param_group_)
    return param_config

def test_sgd_param_groups(SGD):
    test_cases = [
        (
            [{'params': "base"}, {'params': "classifier", 'lr': 1e-3}],
            dict(lr=1e-2, momentum=0.0),
        ),
        (
            [{'params': "base"}, {'params': "classifier"}],
            dict(lr=1e-2, momentum=0.9),
        ),
        (
            [{'params': "base", "lr": 1e-2, "momentum": 0.95}, {'params': "classifier", 'lr': 1e-3}],
            dict(momentum=0.9, weight_decay=0.1),
        ),
    ]
    for description, kwargs in test_cases:
        t.manual_seed(819)

        model = Net2()
        param_config = construct_param_config_from_description(description, model)
        opt = optim.SGD(param_config, **kwargs)
        _train_with_opt(model, opt)
        w0_correct = model.base[0].weight
        
        t.manual_seed(819)
        model = Net2()
        param_config = construct_param_config_from_description(description, model)
        opt = SGD(param_config, **kwargs)
        _train_with_opt(model, opt)
        w0_submitted = model.base[0].weight

        print("\nTesting configuration: ", description)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)

    print("\nTesting that your function doesn't allow duplicates (this should raise an error): ")
    description, kwargs = (
        [{'params': "base", "lr": 1e-2, "momentum": 0.95}, {'params': "base", 'lr': 1e-3}],
        dict(momentum=0.9, weight_decay=0.1),
    )
    try:
        model = Net2()
        param_config = construct_param_config_from_description(description, model)
        opt = SGD(param_config, **kwargs)
    except:
        print("Got an error, as expected.\n")
    else:
        raise Exception("Should have gotten an error from using duplicate parameters, but didn't.")
    

    print("All tests in `test_sgd_param_groups` passed!")

def format_name(name):
    return name.replace("(", "<br>   ").replace(")", "").replace(", ", "<br>   ")

def format_config(config, line_breaks=False):
    if isinstance(config, dict):
        if line_breaks:
            s = "<br>   " + "<br>   ".join([f"{key}={value}" for key, value in config.items()])
        else:
            s = ", ".join([f"{key}={value}" for key, value in config.items()])
    else:
        param_config, args_config = config
        s = "[" + ", ".join(["{" + format_config(param_group_config) + "}" for param_group_config in param_config]) + "], " + format_config(args_config)
    return s

def plot_fn(fn: Callable, x_range=[-2, 2], y_range=[-1, 3], n_points=100, log_scale=True, show_min=False):
    """Plot the specified function over the specified domain.

    If log_scale is True, take the logarithm of the output before plotting.
    """
    x = t.linspace(*x_range, n_points)
    xx = repeat(x, "w -> h w", h=n_points)
    y = t.linspace(*y_range, n_points)
    yy = repeat(y, "h -> h w", w=n_points)

    z = fn(xx, yy)

    max_contour_label = int(z.log().max().item()) + 1
    contour_range = list(range(max_contour_label))

    fig = make_subplots(
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        rows=1, cols=2,
        subplot_titles=["3D plot", "2D log plot"]
    ).update_layout(height=700, width=1600, title_font_size=40).update_annotations(font_size=20)

    fig.add_trace(
        go.Surface(
            x=x, y=y, z=z,
            colorscale="greys",
            showscale=False,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{z:.2f}</b>',
            contours = dict(
                x = dict(show=True, color="grey", start=x_range[0], end=x_range[1], size=0.2),
                y = dict(show=True, color="grey", start=y_range[0], end=y_range[1], size=0.2),
                # z = dict(show=True, color="red", size=0.001)
            )
        ), row=1, col=1
    )
    fig.add_trace(
        go.Contour(
            x=x, y=y, z=t.log(z) if log_scale else z,
            customdata=z,
            hovertemplate = '<b>x</b> = %{x:.2f}<br><b>y</b> = %{y:.2f}<br><b>z</b> = %{customdata:.2f}</b>',
            colorscale="greys",
            # colorbar=dict(tickmode="array", tickvals=contour_range, ticktext=[f"{math.exp(i):.0f}" for i in contour_range])
        ),
        row=1, col=2
    )
    fig.update_traces(showscale=False, col=2)
    if show_min:
        fig.add_trace(
            go.Scatter(
                mode="markers", x=[1.0], y=[1.0], marker_symbol="x", marker_line_color="midnightblue", marker_color="lightskyblue",
                marker_line_width=2, marker_size=12, name="Global minimum"
            ),
            row=1, col=2
        )

    return fig

def plot_optimization_sgd(opt_fn_with_sgd: Callable, fn: Callable, xy: t.Tensor, x_range=[-2, 2], y_range=[-1, 3], lr=0.001, momentum=0.98, n_iters=100, log_scale=True, n_points=100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    xys = opt_fn_with_sgd(fn, xy, lr, momentum, n_iters)
    x, y = xys.T
    z = fn(x, y)
    fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color="red"), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color="red"), line=dict(width=1, color="red")), row=1, col=2)

    fig.update_layout(showlegend=False)
    fig.data = fig.data[::-1]

    return fig

def plot_optimization(opt_fn: Callable, fn: Callable, xy: t.Tensor, optimizers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer) in enumerate(zip(px.colors.qualitative.Set1, optimizers)):
        xys = opt_fn(fn, xy.clone().detach().requires_grad_(True), *optimizer, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name = format_name(str(optimizer_active))
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig

def plot_optimization_with_schedulers(opt_fn_with_scheduler: Callable, fn: Callable, xy: t.Tensor, optimizers: list, schedulers: list, x_range=[-2, 2], y_range=[-1, 3], n_iters: int = 100, log_scale: bool = True, n_points: int = 100, show_min=False):

    fig = plot_fn(fn, x_range, y_range, n_points, log_scale, show_min)

    for i, (color, optimizer, scheduler) in enumerate(zip(px.colors.qualitative.Set1, optimizers, schedulers)):
        optimizer_active = optimizer[0]([xy.clone().detach().requires_grad_(True)], **optimizer[1])
        name_opt = format_name(str(optimizer_active))
        if len(scheduler) == 0:
            scheduler = (None, dict())
            name = name_opt + "<br>(no scheduler)"
        else:
            scheduler_active = scheduler[0](optimizer_active, **scheduler[1])
            name_sch = format_name(str(scheduler_active))
            name = name_opt + "<br>" + name_sch
        xys = opt_fn_with_scheduler(fn, xy.clone().detach().requires_grad_(True), *optimizer, *scheduler, n_iters).numpy()
        x, y = xys.T
        z = fn(x, y)
        fig.add_trace(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(width=6, color=color), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", marker=dict(size=6, color=color), line=dict(width=1, color=color), name=name), row=1, col=2)

    fig.data = fig.data[::-1]

    return fig

def test_sgd(SGD):

    test_cases = [
        dict(lr=0.1, momentum=0.0, weight_decay=0.0),
        dict(lr=0.1, momentum=0.7, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.0),
        dict(lr=0.1, momentum=0.5, weight_decay=0.05),
        dict(lr=0.2, momentum=0.8, weight_decay=0.05),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)


def test_rmsprop(RMSprop):

    test_cases = [
        dict(lr=0.1, alpha=0.9, eps=0.001, weight_decay=0.0, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.5),
        dict(lr=0.1, alpha=0.95, eps=0.0001, weight_decay=0.05, momentum=0.0),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = RMSprop(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)

def test_adam(Adam):

    test_cases = [
        dict(lr=0.1, betas=(0.8, 0.95), eps=0.001, weight_decay=0.0),
        dict(lr=0.1, betas=(0.8, 0.9), eps=0.001, weight_decay=0.05),
        dict(lr=0.2, betas=(0.9, 0.95), eps=0.01, weight_decay=0.08),
    ]
    for opt_config in test_cases:
        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_correct = model.layers[0].weight

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = Adam(model.parameters(), **opt_config)
        _train_with_opt(model, opt)
        w0_submitted = model.layers[0].weight

        print("\nTesting configuration: ", opt_config)
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)

def get_sgd_optimizer(model, opt_config, SGD):
    if isinstance(opt_config, dict):
        return SGD(model.parameters(), **opt_config)
    else:
        opt_params = [d.copy() for d in opt_config[0]]
        _opt_config = opt_config[1]
        weight_params = [param for name, param in model.named_parameters() if "weight" in name]
        bias_params = [param for name, param in model.named_parameters() if "bias" in name]
        for param_group in opt_params:
            param_group["params"] = weight_params if param_group["params"] == "weights" else bias_params
        return SGD(opt_params, **_opt_config)

def test_ExponentialLR(ExponentialLR, SGD):

    print("Testing ExponentialLR, training loop has 30 epochs, 4 batches per epoch")


    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = ExponentialLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_ExponentialLR` passed!")

def test_StepLR(StepLR, SGD):

    print("Testing StepLR, training loop has 30 epochs, 4 batches per epoch")

    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=30, gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=3, gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(step_size=1, gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(step_size=3, gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = StepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.StepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_StepLR` passed!")


def test_MultiStepLR(MultiStepLR, SGD):

    print("Testing MultiStepLR, training loop has 30 epochs, 4 batches per epoch")

    test_cases = [
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(milestones=[40], gamma=1.0)),
        dict(opt_config=dict(lr=0.01, momentum=0.0, weight_decay=0.0), scheduler_config=dict(milestones=[10], gamma=0.5)),
        dict(opt_config=dict(lr=0.01, momentum=0.9, weight_decay=0.1), scheduler_config=dict(milestones=[10, 15], gamma=0.5)),
    ]
    for config in test_cases:
        opt_config = config["opt_config"].copy()
        scheduler_config = config["scheduler_config"]

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = SGD(model.parameters(), **opt_config)
        scheduler = MultiStepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_correct = model.layers[0].weight
        b0_correct = model.layers[0].bias

        t.manual_seed(819)
        model = Net(2, 32, 2)
        opt = t.optim.SGD(model.parameters(), **opt_config)
        scheduler = optim.lr_scheduler.MultiStepLR(opt, **scheduler_config)
        _train_with_scheduler(model, opt, scheduler)
        w0_submitted = model.layers[0].weight
        b0_submitted = model.layers[0].bias

        print("\nTesting configuration:\n\toptimizer: ", format_config(opt_config), "\n\tscheduler: ", format_config(scheduler_config))
        assert isinstance(w0_correct, t.Tensor)
        assert isinstance(w0_submitted, t.Tensor)
        assert isinstance(b0_correct, t.Tensor)
        assert isinstance(b0_submitted, t.Tensor)
        t.testing.assert_close(w0_correct, w0_submitted, rtol=0, atol=1e-5)
        t.testing.assert_close(b0_correct, b0_submitted, rtol=0, atol=1e-5)
    print("\nAll tests in `test_MultiStepLR` passed!")

def test_random_mask(random_mask, input_size=10000, max_seq=128):
    print("Testing empirical frequencies")
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

    input_ids = t.randint(0, 1_000_000, (input_size, max_seq))
    select_frac = 0.85
    mask_frac = 0.75
    random_frac = 0.10

    masked, was_selected = random_mask(
        input_ids, tokenizer.mask_token_id, 1_000_000, select_frac, mask_frac, random_frac
    )

    print('Checking fraction of tokens selected...')
    actual_selected_frac = was_selected.float().mean().item()
    t.testing.assert_close(actual_selected_frac, select_frac, atol=1e-5, rtol=0)

    print('Checking fraction of tokens masked...')
    actual_mask_frac = (masked == tokenizer.mask_token_id).float().mean().item()
    expected_mask_frac = select_frac * mask_frac
    t.testing.assert_close(actual_mask_frac, expected_mask_frac, atol=1e-5, rtol=0)

    print('Checking fraction of tokens masked OR randomized...')
    changed_frac = (masked != input_ids).float().mean().item()
    expected_changed_frac = select_frac * (mask_frac + random_frac)
    t.testing.assert_close(changed_frac, expected_changed_frac, atol=1e-5, rtol=0)


def flat(x: t.Tensor) -> t.Tensor:
    """Helper function for combining batch and sequence dimensions."""
    return rearrange(x, "b s ... -> (b s) ...")


def unflat(x: t.Tensor, max_seq: int) -> t.Tensor:
    """Helper function for separating batch and sequence dimensions."""
    return rearrange(x, "(b s) ... -> b s ...", s=max_seq)


def test_cross_entropy_selected(cross_entropy_selected, verbose=False):
    t.manual_seed(0)

    shape = (3, 4, 10)
    pred = t.randn(*shape)
    y = t.randint(0, 10, shape[:-1])

    # none selected
    #selected = t.zeros(shape[:-1], dtype=t.int)
    #theirs = cross_entropy_selected(pred, y, selected)
    #assert theirs.isnan().all()

    # all selected
    selected = t.ones(shape[:-1], dtype=t.int)
    theirs = cross_entropy_selected(pred, y, selected)
    ours = F.cross_entropy(flat(pred), flat(y))
    if verbose:
        print(theirs, ours)
    assert theirs == ours

    # some selected
    selected = (t.rand(shape[:-1]) > 0.5).int()
    theirs = cross_entropy_selected(pred, y, selected)
    s_pred = flat(pred)[flat(selected).bool()]
    s_y = flat(y)[flat(selected).bool()]
    ours = F.cross_entropy(s_pred, s_y)
    if verbose:
        print(theirs, ours)
    assert theirs == ours