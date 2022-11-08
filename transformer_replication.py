#%% 
import transformers
import torch as t
import torch.nn as nn
import utils_w1d1
from typing import Union, List
from fancy_einsum import einsum
# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
print(tokenizer("hello meg"))
print(tokenizer.encode("hello meg"))
print(tokenizer.decode([31373, 17243]))
print(tokenizer.tokenize("hello meg"))
print(f"'{tokenizer.decode(17243)}'")
# %%
class Embedding(nn.Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(t.randn((self.num_embeddings, self.embedding_dim)))

    def forward(self, x: t.LongTensor) -> t.Tensor:
        '''For each integer in the input, return that row of the embedding.
        '''
        #return einsum('num_embeddings embedding_dim, i num_embeddings -> i embedding_dim', self.weight, nn.functional.one_hot(x, num_classes=self.num_embeddings).float())
        return self.weight[x]

    def extra_repr(self) -> str:
        return f"{self.num_embeddings}, {self.embedding_dim}"

utils_w1d1.test_embedding(Embedding)
# %%
#TODO positional encoding
# %%
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(self.normalized_shape))
            self.bias = nn.Parameter(t.zeros(self.normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        normalized_shape_dims = 1 if isinstance(self.normalized_shape, int) else len(self.normalized_shape)
        x_mean = x.mean(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True)
        x_var = x.var(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True)
        print(x.var())
        x_scaled = (x - x_mean) / t.sqrt(x_var + self.eps)
        print(x_scaled.var())
        if self.elementwise_affine:
            return x_scaled * self.weight + self.bias
        print(self.bias.mean())
        return x_scaled

    def extra_repr(self) -> str:
        pass

utils_w1d1.test_layernorm_mean_1d(LayerNorm)
utils_w1d1.test_layernorm_mean_2d(LayerNorm)
utils_w1d1.test_layernorm_std(LayerNorm)
utils_w1d1.test_layernorm_exact(LayerNorm)
utils_w1d1.test_layernorm_backward(LayerNorm)
# %%
