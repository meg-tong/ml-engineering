#%%
import functools
import json
import os
from typing import Any, List, Tuple, Union, Optional
import matplotlib.pyplot as plt
import torch as t
import torch.nn.functional as F
from fancy_einsum import einsum
from sklearn.linear_model import LinearRegression
from torch import nn
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from einops import rearrange, repeat
import pandas as pd
import numpy as np

import interpretability_utils
from interpretability_utils import DataSet
from reference_transformer import ParenTransformer, SimpleTokenizer

MAIN = __name__ == "__main__"
DEVICE = t.device("cpu")

if MAIN:
    model = ParenTransformer(ntoken=5, nclasses=2, d_model=56, nhead=2, d_hid=56, nlayers=3).to(DEVICE)
    state_dict = t.load("reference_state_dict.pt")
    model.to(DEVICE)
    model.load_simple_transformer_state_dict(state_dict)
    model.eval()
    tokenizer = SimpleTokenizer("()")
    with open("reference_data.json") as f:
        data_tuples: List[Tuple[str, bool]] = json.load(f)
        print(f"Loaded {len(data_tuples)} examples")
    assert isinstance(data_tuples, list)

    N_SAMPLES = 5000
    data_tuples = data_tuples[:N_SAMPLES]
    data = DataSet(data_tuples)
    
    fig = px.histogram([len(x[0]) for x in data])
    fig.show()
# %%
def is_balanced_forloop(parens: str) -> bool:
    '''Return True if the parens are balanced.

    Parens is just the ( and ) characters, no begin or end tokens.
    '''
    elevation = 0
    for p in parens:
        elevation += (1 if p == "(" else -1)
        if elevation < 0:
            return False
    return elevation == 0

if MAIN:
    examples = ["()", "))()()()()())()(())(()))(()(()(()(", "((()()()()))", "(()()()(()(())()", "()(()(((())())()))"]
    labels = [True, False, True, False, True]
    for (parens, expected) in zip(examples, labels):
        actual = is_balanced_forloop(parens)
        assert expected == actual, f"{parens}: expected {expected} got {actual}"
    print("is_balanced_forloop ok!")
# %%
def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    '''Return probability that each example is balanced'''
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

if MAIN:
    toks = tokenizer.tokenize(examples).to(DEVICE)
    out = model(toks)
    prob_balanced = out.exp()[:, 1]
    print("Model confidence:\n" + "\n".join([f"{ex:34} : {prob:.4%}" for ex, prob in zip(examples, prob_balanced)]))

    test_set = data
    n_correct = t.sum((run_model_on_data(model, test_set).argmax(-1) == test_set.isbal).int())
    print(f"\nModel got {n_correct} out of {len(data)} training examples correct!")
# %%
def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    return model.decoder.weight[0] - model.decoder.weight[1]
# %%
def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    inputs = []
    def fn(module, input, output):
        inputs.append(input[0])
        return None
    handle = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    handle.remove()
    return t.concat(inputs).detach()

def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    outputs = []
    def fn(module, input, output):
        outputs.append(output)
        return None
    handle = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    handle.remove()
    return t.concat(outputs).detach()

if MAIN:
    interpretability_utils.test_get_inputs(get_inputs, model, data)
    interpretability_utils.test_get_outputs(get_outputs, model, data)
# %%
def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit for all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in torch.tensor() to make a dimensionless tensor)
    '''
    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)
    if seq_pos is not None:
        inputs = inputs[:, seq_pos, :].squeeze()
        outputs = outputs[:, seq_pos, :].squeeze()
    else:
        inputs = rearrange(inputs, "batch seq emb -> batch (seq emb)")
        outputs = rearrange(outputs, "batch seq emb -> batch (seq emb)")
    regr = LinearRegression()
    regr.fit(inputs, outputs)
    return regr, t.tensor(regr.score(inputs, outputs))

if MAIN:
    (final_ln_fit, r2) = get_ln_fit(model, data, model.norm, seq_pos=0)
    print("r^2: ", r2)
    interpretability_utils.test_final_ln_fit(model, data, get_ln_fit)
# %%
def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    regr, _ = get_ln_fit(model, data, model.norm, seq_pos=0)
    return t.matmul(get_post_final_ln_dir(model), t.tensor(regr.coef_))

if MAIN:
    interpretability_utils.test_pre_final_ln_dir(model, data, get_pre_final_ln_dir)
# %%
