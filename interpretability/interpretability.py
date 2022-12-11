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
    data = interpretability_utils.DataSet(data_tuples)
    
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
