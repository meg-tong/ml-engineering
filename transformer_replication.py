#%% 
import transformers
import torch as t
import torch.nn as nn
import utils_w1d1
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
