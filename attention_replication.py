# %%
import torch as t
import torch.nn as nn
import utils_w1d1
from typing import Union, List, Optional
from fancy_einsum import einsum
from einops import repeat, rearrange, reduce
import numpy as np
#%%
def single_head_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (batches x seq_Q x head_size)
    K: shape (batches x seq_K x head_size)
    V: shape (batches x seq_K x head_size)

    Return: shape (batches x seq_Q x head_size)
    '''
    
    attention_scores = einsum('batches seq_Q head_size, batches seq_K head_size -> batches seq_Q seq_K', Q, K)
    #Ignore masking
    attention_probabilities = nn.functional.softmax(attention_scores / np.sqrt(Q.shape[-1]), dim=2)
    attention_values = einsum('batches seq_Q seq_K, batches seq_K head_size -> batches seq_Q head_size', attention_probabilities, V)
    return attention_values

def test_single_head_attention_shape(single_head_attention):
    Q = t.randn(1, 3, 2)
    K = t.randn(1, 5, 2)
    V = t.randn(1, 5, 2)
    attention_values = single_head_attention(Q, K, V)
    assert Q.shape == attention_values.shape
    print(f"All tests in `test_single_head_attention_shape` passed.")

def test_single_head_attention(single_head_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = single_head_attention(Q.float(), K.float(), V.float())
    t.testing.assert_close(attention_values, t.tensor([[[9.7880e-04, 9.9902e-01, 9.7880e-04], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_single_head_attention` passed.")

if __name__ == "__main__":
    test_single_head_attention_shape(single_head_attention)
    test_single_head_attention(single_head_attention)
# %%
def single_head_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor) -> t.Tensor:
    '''
    Should return the results of masked self-attention.

    See "The Decoder Side" section of the Illustrated Transformer for an explanation of masking.

    Q: shape (batches x seq_Q x head_size)
    K: shape (batches x seq_K x head_size)
    V: shape (batches x seq_K x head_size)

    Return: shape (batches x seq_Q x head_size)
    '''
    attention_scores = einsum('batches seq_Q head_size, batches seq_K head_size -> batches seq_Q seq_K', Q, K)
    batches, seq_Q, head_size = Q.shape
    batches, seq_K, head_size = K.shape

    q_index = repeat(t.arange(0, seq_Q), 'q -> b q k', b=batches, k=seq_K)
    k_index = repeat(t.arange(0, seq_K), 'k -> b q k', b=batches, q=seq_Q)
    mask = k_index <= q_index
    attention_scores = t.where(mask, attention_scores, -t.inf)
    attention_probabilities = nn.functional.softmax(attention_scores / np.sqrt(Q.shape[-1]), dim=2)
    attention_values = einsum('batches seq_Q seq_K, batches seq_K head_size -> batches seq_Q head_size', attention_probabilities, V)
    return attention_values

def test_single_head_masked_attention(single_head_masked_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = single_head_masked_attention(Q.float(), K.float(), V.float())
    t.testing.assert_close(attention_values, t.tensor([[[1, 0, 1], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_single_head_attention` passed.")

if __name__ == "__main__":
    test_single_head_attention_shape(single_head_masked_attention)
    test_single_head_masked_attention(single_head_masked_attention)
# %%
def multihead_masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int, additive_attention_mask: Optional[t.Tensor] = None):
    '''
    Implements multihead masked attention on the matrices Q, K and V.

    Q: shape (batch, seq, nheads*headsize)
    K: shape (batch, seq, nheads*headsize)
    V: shape (batch, seq, nheads*headsize)

    returns: shape (batch, seq, nheads*headsize)
    '''
    new_Q = rearrange(Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_K = rearrange(K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
    new_V = rearrange(V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)

    attention_scores = einsum('batches nheads seq_Q head_size, batches nheads seq_K head_size -> batches nheads seq_Q seq_K', new_Q, new_K)
    batches, _, seq_Q, head_size = new_Q.shape
    batches, _, seq_K, head_size = new_K.shape
    if additive_attention_mask is None:
        q_index = repeat(t.arange(0, seq_Q), 'seq_Q -> batches nheads seq_Q seq_K', batches=batches, seq_K=seq_K, nheads=num_heads)
        k_index = repeat(t.arange(0, seq_K), 'seq_K -> batches nheads seq_Q seq_K', batches=batches, seq_Q=seq_Q, nheads=num_heads)
        mask = k_index <= q_index
        masked_attention_scores = t.where(mask, attention_scores, -t.inf)
    else:
        masked_attention_scores = additive_attention_mask
    attention_probabilities = nn.functional.softmax(masked_attention_scores / np.sqrt(head_size), dim=-1)
    attention_values = einsum('batches nheads seq_Q seq_K, batches nheads seq_K head_size -> batches seq_Q nheads head_size', attention_probabilities, new_V)
    return rearrange(attention_values, 'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)')

def test_multihead_masked_attention(multihead_masked_attention):
    Q = t.tensor([[[7, 4, 1], [6, 3, 0], [5, 2, 1]]])
    K = t.tensor([[[1, 3, 5], [2, 4, 6]]])
    V = t.tensor([[[1, 0, 1], [0, 1, 0]]])
    attention_values = multihead_masked_attention(Q.float(), K.float(), V.float(), num_heads=1)
    t.testing.assert_close(attention_values, t.tensor([[[1, 0, 1], [5.5073e-03, 9.9449e-01, 5.5073e-03], [9.7682e-03, 9.9023e-01, 9.7682e-03]]]), rtol=0.01, atol=0.001)
    print(f"All tests in `test_multihead_masked_attention` passed.")  

if __name__ == "__main__":
    test_multihead_masked_attention(multihead_masked_attention)
# %%
class MultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = multihead_masked_attention(Q, K, V, self.num_heads, additive_attention_mask)
        return self.W_O(attention_values)
# %%
def test_MultiheadMaskedAttention_shape(MultiheadMaskedAttention):
    mma = MultiheadMaskedAttention(1, 1)
    x = t.randn(2, 7, 1)
    output = mma.forward(x)
    assert x.shape == output.shape
    print(f"All tests in `test_MultiheadMaskedAttention_shape` passed.")

if __name__ == "__main__":
    test_MultiheadMaskedAttention_shape(MultiheadMaskedAttention)
# %%
