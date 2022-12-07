#%%
import numpy as np
import torch as t
import torch.nn as nn
import transformers
from einops import rearrange, repeat
from fancy_einsum import einsum

import transformers_utils
import transformer_replication


#%%
class GPTMultiheadMaskedAttention(nn.Module):
    W_QKV: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int, p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.W_QKV = nn.Linear(hidden_size, 3 * hidden_size, bias=True)
        self.W_O = nn.Linear(hidden_size, hidden_size, bias=True)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)

    def forward(self, x: t.Tensor) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        QKV = self.W_QKV(x)
        Q = QKV[..., :self.hidden_size]
        K = QKV[..., self.hidden_size:-self.hidden_size]
        V = QKV[..., -self.hidden_size:]
        attention_values = self.multihead_masked_attention(Q, K, V, self.num_heads)
        out = self.W_O(attention_values)
        return self.dropout2(out)

    def multihead_masked_attention(self, Q: t.Tensor, K: t.Tensor, V: t.Tensor, num_heads: int):
        new_Q = rearrange(Q, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
        new_K = rearrange(K, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)
        new_V = rearrange(V, 'batch seq (nheads headsize) -> batch nheads seq headsize', nheads=num_heads)

        attention_scores = einsum('batches nheads seq_Q head_size, batches nheads seq_K head_size -> batches nheads seq_Q seq_K', new_Q, new_K)
        batches, _, seq_Q, head_size = new_Q.shape
        batches, _, seq_K, head_size = new_K.shape
        q_index = repeat(t.arange(0, seq_Q), 'seq_Q -> batches nheads seq_Q seq_K', batches=batches, seq_K=seq_K, nheads=num_heads)
        k_index = repeat(t.arange(0, seq_K), 'seq_K -> batches nheads seq_Q seq_K', batches=batches, seq_Q=seq_Q, nheads=num_heads)
        mask = k_index <= q_index
        masked_attention_scores = t.where(mask, attention_scores, -t.inf)
        attention_probabilities = nn.functional.softmax(masked_attention_scores / np.sqrt(head_size), dim=-1)
        attention_probabilities = self.dropout1(attention_probabilities) # not in original transformer
        attention_values = einsum('batches nheads seq_Q seq_K, batches nheads seq_K head_size -> batches seq_Q nheads head_size', attention_probabilities, new_V)
        return rearrange(attention_values, 'batches seq_Q nheads head_size -> batches seq_Q (nheads head_size)')

class GPTDecoderBlock(nn.Module):

    def __init__(self, config: transformer_replication.TransformerConfig):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.attention = GPTMultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = transformer_replication.MLP(config)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.layer_norm1(x)
        y = self.attention(y)
        x = x + y
        z = self.layer_norm2(x)
        z = self.mlp(z)
        x = x + z
        return x

config = transformer_replication.TransformerConfig(
        num_layers=12,
        num_heads=12,
        vocab_size=50257,
        hidden_size=768,
        max_seq_len=1024,
        dropout=0.1,
        layer_norm_epsilon=1e-5
    )

#%%
def copy_weights(my_model, model, gpt2=True):
    '''
    Copy over the weights from the official model to your implementation of the model.
    Returns your model, with weights loaded in.
    '''
    named_parameters = {}
    for (name, param), (my_name, my_param) in zip(model.named_parameters(), my_model.named_parameters()):
        if gpt2 and len(param.shape) > 1 and param.shape[0] == my_param.shape[1] and param.shape[1] == my_param.shape[0]:
            named_parameters[my_name] = rearrange(param, 'i j -> j i')
        else:
            named_parameters[my_name] = param

    my_model.load_state_dict(named_parameters)
    return my_model
#%%
if __name__ == "__main__":
    my_gpt = transformer_replication.DecoderOnlyTransformer(config, transformer_replication.Embedding, GPTDecoderBlock).train()
    gpt = transformers.AutoModelForCausalLM.from_pretrained("gpt2").train()
    #transformers_utils.print_param_count(my_gpt, gpt, use_state_dict=False)
    my_gpt = copy_weights(my_gpt, gpt)
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    transformers_utils.test_load_pretrained_weights(gpt, tokenizer)
    transformers_utils.test_load_pretrained_weights(my_gpt, tokenizer)

# %%
# %%
