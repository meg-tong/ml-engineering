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
from torch import nn
from torchvision import datasets, transforms
from tqdm.auto import tqdm
import random

Arr = np.ndarray

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