from typing import Any, List, Tuple, Union

import torch as t
import torch.nn.functional as F
from einops import rearrange
from fancy_einsum import einsum
from reference_transformer import ParenTransformer, SimpleTokenizer
from sklearn.linear_model import LinearRegression
from torch import nn

DEVICE = t.device("cpu")
tokenizer = SimpleTokenizer("()")

class DataSet:
    """A dataset containing sequences, is_balanced labels, and tokenized sequences"""

    def __init__(self, data_tuples: list):
        """
        data_tuples is List[Tuple[str, bool]] signifying sequence and label
        """
        self.strs = [x[0] for x in data_tuples]
        self.isbal = t.tensor([x[1] for x in data_tuples]).to(device=DEVICE, dtype=t.bool)
        self.toks = tokenizer.tokenize(self.strs).to(DEVICE)
        self.open_proportion = t.tensor([s.count("(") / len(s) for s in self.strs])
        self.starts_open = t.tensor([s[0] == "(" for s in self.strs]).bool()

    def __len__(self) -> int:
        return len(self.strs)

    def __getitem__(self, idx) -> Union["DataSet", Tuple[str, t.Tensor, t.Tensor]]:
        if type(idx) == slice:
            return self.__class__(list(zip(self.strs[idx], self.isbal[idx])))
        return (self.strs[idx], self.isbal[idx], self.toks[idx])

    @property
    def seq_length(self) -> int:
        return self.toks.size(-1)

    @classmethod
    def with_length(cls, data_tuples: List[Tuple[str, bool]], selected_len: int) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if len(s) == selected_len])

    @classmethod
    def with_start_char(cls, data_tuples: List[Tuple[str, bool]], start_char: str) -> "DataSet":
        return cls([(s, b) for (s, b) in data_tuples if s[0] == start_char])

def run_model_on_data(model: ParenTransformer, data: DataSet, batch_size: int = 200) -> t.Tensor:
    """Return probability that each example is balanced"""
    ln_probs = []
    for i in range(0, len(data.strs), batch_size):
        toks = data.toks[i : i + batch_size]
        with t.no_grad():
            out = model(toks)
        ln_probs.append(out)
    out = t.cat(ln_probs).exp()
    assert out.shape == (len(data), 2)
    return out

def get_post_final_ln_dir(model: ParenTransformer) -> t.Tensor:
    return model.decoder.weight[0] - model.decoder.weight[1]

def get_inputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the inputs to a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    acts = []
    fn = lambda module, inputs, output: acts.append(inputs[0].detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()


def get_outputs(model: ParenTransformer, data: DataSet, module: nn.Module) -> t.Tensor:
    '''
    Get the outputs from a particular submodule of the model when run on the data.
    Returns a tensor of size (data_pts, seq_pos, emb_size).
    '''
    acts = []
    fn = lambda module, inputs, output: acts.append(output.detach().clone())
    h = module.register_forward_hook(fn)
    run_model_on_data(model, data)
    h.remove()
    out = t.concat(acts, dim=0)
    assert out.shape == (len(data), data.seq_length, model.d_model)
    return out.clone()


# %%

def get_ln_fit(
    model: ParenTransformer, data: DataSet, ln_module: nn.LayerNorm, seq_pos: Union[None, int]
) -> Tuple[LinearRegression, t.Tensor]:
    '''
    if seq_pos is None, find best fit aggregated over all sequence positions. Otherwise, fit only for given seq_pos.

    Returns: A tuple of a (fitted) sklearn LinearRegression object and a dimensionless tensor containing the r^2 of the fit (hint: wrap a value in t.tensor() to make a dimensionless tensor)
    '''
    inputs = get_inputs(model, data, ln_module)
    outputs = get_outputs(model, data, ln_module)

    if seq_pos is None:
        inputs = rearrange(inputs, "batch seq hidden -> (batch seq) hidden")
        outputs = rearrange(outputs, "batch seq hidden -> (batch seq) hidden")
    else:
        inputs = inputs[:, seq_pos, :]
        outputs = outputs[:, seq_pos, :]

    final_ln_fit = LinearRegression().fit(inputs, outputs)

    r2 = t.tensor(final_ln_fit.score(inputs, outputs))

    return (final_ln_fit, r2)

# %%

def get_pre_final_ln_dir(model: ParenTransformer, data: DataSet) -> t.Tensor:
    post_final_ln_dir = get_post_final_ln_dir(model)
    L = t.from_numpy(get_ln_fit(model, data, model.norm, seq_pos=0)[0].coef_)

    return t.einsum("i,ij->j", post_final_ln_dir, L)

# %%

def get_out_by_head(model: ParenTransformer, data: DataSet, layer: int) -> t.Tensor:
    '''

    Get the output of the heads in a particular layer when the model is run on the data.
    Returns a tensor of shape (batch, nheads, seq, emb)
    '''
    # Capture the inputs to this layer using the input hook function you wrote earlier - call the inputs `r`
    module = model.layers[layer].self_attn.W_O
    r = get_inputs(model, data, module)

    # Reshape the inputs so that heads go along their own dimension (see expander above)
    r = rearrange(r, "batch seq (nheads headsize) -> batch nheads seq headsize", nheads=model.nhead)

    # Extract the weights from the model directly, and reshape them so that heads go along their own dimension
    W_O = module.weight
    W_O = rearrange(W_O, "emb (nheads headsize) -> nheads emb headsize", nheads=model.nhead)

    # Perform the matrix multiplication shown in the expander above (but keeping the terms in the sum separate)
    out_by_heads = einsum("batch nheads seq headsize, nheads emb headsize -> batch nheads seq emb", r, W_O)
    return out_by_heads

# %%

def get_out_by_components(model: ParenTransformer, data: DataSet) -> t.Tensor:
    '''
    Computes a tensor of shape [10, dataset_size, seq_pos, emb] representing the output of the model's components when run on the data.
    The first dimension is  [embeddings, head 0.0, head 0.1, mlp 0, head 1.0, head 1.1, mlp 1, head 2.0, head 2.1, mlp 2]
    '''
    # Get the outputs of each head, for each layer
    head_outputs = [get_out_by_head(model, data, layer) for layer in range(model.nlayers)]
    # Get the MLP outputs for each layer
    mlp_outputs = [get_outputs(model, data, model.layers[layer].linear2) for layer in range(model.nlayers)]
    # Get the embedding outputs (note that model.pos_encoder is applied after the token embedding)
    embedding_output = get_outputs(model, data, model.pos_encoder)

    # Start with [embeddings]
    out = [embedding_output]
    for layer in range(model.nlayers):
        # Append [head n.0, head n.1, mlp n] for n = 0,1,2
        # Note that the heads are in the second dimension of the head_outputs tensor (the first is batch)
        out.extend([head_outputs[layer][:, 0], head_outputs[layer][:, 1], mlp_outputs[layer]])

    return t.stack(out, dim=0)

def get_WO(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_O matrix of a head. Should be a CPU tensor of size (d_model, d_model / num_heads)
    '''
    # Get W_O which has shape (embed_size, nheads * headsize)
    W_O = model.layers[layer].self_attn.W_O.weight
    # Reshape it to (embed_size, nheads, headsize) and get the correct head
    W_O_head = rearrange(W_O, "embed_size (nheads headsize) -> embed_size nheads headsize", nheads=model.nhead)[:, head]
    
    return W_O_head.detach().clone()


def get_WV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    '''
    Returns the W_V matrix of a head. Should be a CPU tensor of size (d_model / num_heads, d_model)
    '''
    # Get W_V which has shape (nheads * headsize, embed_size)
    W_V = model.layers[layer].self_attn.W_V.weight
    # Reshape it to (nheads, headsize, embed_size) and get the correct head
    W_V_head = rearrange(W_V, "(nheads headsize) embed_size -> nheads headsize embed_size", nheads=model.nhead)[head]

    return W_V_head.detach().clone()

def get_pre_20_dir(model, data):
    '''
    Returns the direction propagated back through the OV matrix of 2.0 and then through the layernorm before the layer 2 attention heads.
    '''
    pre_final_ln_dir = get_pre_final_ln_dir(model, data)
    WOV = get_WOV(model, 2, 0)

    # Get the regression fit for the second layernorm in layer 2
    layer2_ln_fit = get_ln_fit(model, data, model.layers[2].norm1, seq_pos=1)[0]
    layer2_ln_coefs = t.from_numpy(layer2_ln_fit.coef_)

    # Propagate back through the layernorm
    pre_20_dir = t.einsum("i,ij,jk->k", pre_final_ln_dir, WOV, layer2_ln_coefs)

    return pre_20_dir

def get_WOV(model: ParenTransformer, layer: int, head: int) -> t.Tensor:
    return get_WO(model, layer, head) @ get_WV(model, layer, head)

def embedding(model: ParenTransformer, tokenizer: SimpleTokenizer, char: str) -> t.Tensor:
    assert char in ("(", ")")
    input_id = tokenizer.t_to_i[char]
    input = t.tensor([input_id]).to(DEVICE)
    return model.encoder(input).clone()


def get_Q_and_K(model: ParenTransformer, layer: int, head: int) -> Tuple[t.Tensor, t.Tensor]:
    """
    Get the Q and K weight matrices for the attention head at the given indices.
    Return: Tuple of two tensors, both with shape (embedding_size, head_size)
    """
    q_proj: nn.Linear = model.layers[layer].self_attn.W_Q
    k_proj: nn.Linear = model.layers[layer].self_attn.W_K
    num_heads = model.nhead
    q_mats_by_head = rearrange(q_proj.weight, "(head head_size) out -> out head head_size", head=num_heads)
    k_mats_by_head = rearrange(k_proj.weight, "(head head_size) out -> out head head_size", head=num_heads)
    q_mat = q_mats_by_head[:, head]
    assert q_mat.shape == (model.d_model, model.d_model // model.nhead)
    k_mat = k_mats_by_head[:, head]
    assert k_mat.shape == (model.d_model, model.d_model // model.nhead)
    return q_mat, k_mat


def qk_calc_termwise(
    model: ParenTransformer,
    layer: int,
    head: int,
    q_embedding: t.Tensor,
    k_embedding: t.Tensor,
) -> t.Tensor:
    """
    Get the pre-softmax attention scores that would be calculated by the given attention head from the given embeddings.
    q_embedding: tensor of shape (seq_len, embedding_size)
    k_embedding: tensor of shape (seq_len, embedding_size)
    Returns: tensor of shape (seq_len, seq_len)
    """
    q_mat, k_mat = get_Q_and_K(model, layer, head)
    qs = einsum("i o, x i -> x o", q_mat, q_embedding)
    ks = einsum("i o, y i -> y o", k_mat, k_embedding)
    scores = einsum("x o, y o -> x y", qs, ks)
    return scores.squeeze()

def test_get_inputs(get_inputs, model, data):

    module = model.layers[1].linear2

    expected = get_inputs(model, data, module)
    actual = get_inputs(model, data, module)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_inputs` passed.")

def test_get_outputs(get_outputs, model, data):

    module = model.layers[1].linear2

    expected = get_outputs(model, data, module)
    actual = get_outputs(model, data, module)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_outputs` passed.")

def test_get_out_by_head(get_out_by_head, model, data):

    layer = 2

    expected = get_out_by_head(model, data, layer)
    actual = get_out_by_head(model, data, layer)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_out_by_head` passed.")

def test_get_out_by_component(get_out_by_components, model, data):

    expected = get_out_by_components(model, data)
    actual = get_out_by_components(model, data)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_out_by_component` passed.")

def test_final_ln_fit(model, data, get_ln_fit):

    expected, exp_r2 = get_ln_fit(model, data, model.norm, 0)
    actual, act_r2 = get_ln_fit(model, data, model.norm, 0)

    t.testing.assert_close(t.tensor(actual.coef_), t.tensor(expected.coef_))
    t.testing.assert_close(t.tensor(actual.intercept_), t.tensor(expected.intercept_))
    t.testing.assert_close(act_r2, exp_r2)
    print("All tests in `test_final_ln_fit` passed.")

def test_pre_final_ln_dir(model, data, get_pre_final_ln_dir):

    expected = get_pre_final_ln_dir(model, data)
    actual = get_pre_final_ln_dir(model, data)
    similarity = t.nn.functional.cosine_similarity(actual, expected, dim=0).item()
    t.testing.assert_close(similarity, 1.0)
    print("All tests in `test_pre_final_ln_dir` passed.")

def test_get_WV(model, get_WV):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        v = get_WV(model, layer, head)
        their_v = get_WV(model, layer, head)
        t.testing.assert_close(their_v, v)
    print("All tests in `test_get_WV` passed.")

def test_get_WO(model, get_WO):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        o = get_WO(model, layer, head)
        their_o = get_WO(model, layer, head)
        t.testing.assert_close(their_o, o)
    print("All tests in `test_get_WO` passed.")

def test_get_pre_20_dir(model, data, get_pre_20_dir):

    expected = get_pre_20_dir(model, data)
    actual = get_pre_20_dir(model, data)
    
    t.testing.assert_close(actual, expected)
    print("All tests in `test_get_pre_20_dir` passed.")

def embedding_test(model, tokenizer, embedding_fn):

    open_encoding = embedding(model, tokenizer, "(")
    closed_encoding = embedding(model, tokenizer, ")")

    t.testing.assert_close(embedding_fn(model, tokenizer, "("), open_encoding)
    t.testing.assert_close(embedding_fn(model, tokenizer, ")"), closed_encoding)
    print("All tests in `embedding_test` passed.")

def qk_test(model, their_get_q_and_k):

    indices = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]

    for layer, head in indices:
        q, k = get_Q_and_K(model, layer, head)
        their_q, their_k = their_get_q_and_k(model, layer, head)
        t.testing.assert_close(their_q, q)
        t.testing.assert_close(their_k, k)
    print("All tests in `qk_test` passed.")

def test_qk_calc_termwise(model, tokenizer, their_get_q_and_k):

    embedding = model.encoder(tokenizer.tokenize(["()()()()"]).to(DEVICE)).squeeze()
    expected = qk_calc_termwise(model, 0, 0, embedding, embedding)
    actual = their_get_q_and_k(model, 0, 0, embedding, embedding)

    t.testing.assert_close(actual, expected)
    print("All tests in `test_qk_calc_termwise` passed.")

def remove_hooks(module: t.nn.Module):
    """Remove all hooks from module.
    Use module.apply(remove_hooks) to do this recursively.
    """
    module._backward_hooks.clear()
    module._forward_hooks.clear()
    module._forward_pre_hooks.clear()