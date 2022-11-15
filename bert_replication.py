#%% 
from typing import List, Optional

import torch as t
import transformers
from einops import repeat
from fancy_einsum import einsum
from torch import nn

import arena_utils
import attention_replication
import gpt2_replication
import transformer_replication


# %%
class BiasLayer(nn.Module):
    def __init__(self, length: int):
        super().__init__()
        self.bias = nn.Parameter(t.randn((length)))
    
    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.bias

class BERTMultiheadMaskedAttention(nn.Module):
    W_Q: nn.Linear
    W_K: nn.Linear
    W_V: nn.Linear
    W_O: nn.Linear

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert self.hidden_size % self.num_heads == 0
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.W_O = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)

        Return: shape (batch, seq, hidden_size)
        '''
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        attention_values = attention_replication.multihead_masked_attention(Q, K, V, self.num_heads, additive_attention_mask)
        return self.W_O(attention_values)

class BERTBlock(nn.Module):
    def __init__(self, config: transformer_replication.TransformerConfig):
        super().__init__()
        self.attention = BERTMultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = transformer_replication.MLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(self, x: t.Tensor, additive_attention_mask: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        additive_attention_mask: shape (batch, nheads=1, seqQ=1, seqK)
        '''
        y = self.attention(x, additive_attention_mask)
        x = x + y
        x = self.layer_norm1(x)
        z = self.mlp(x)
        x = x + z
        x = self.layer_norm2(x)
        return x

class BERTCommon(nn.Module):
    def __init__(self, config: transformer_replication.TransformerConfig):
        super().__init__()
        self.token_embedding = transformer_replication.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = transformer_replication.Embedding(config.max_seq_len, config.hidden_size)
        self.token_type_embedding = transformer_replication.Embedding(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[BERTBlock(config) for _ in range(config.num_layers)])
        
    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None, token_type_ids: Optional[t.Tensor] = None) -> t.Tensor:
        '''
        input_ids: (batch, seq) - the token ids
        one_zero_attention_mask: (batch, seq) - only used in training, passed to `make_additive_attention_mask` and used in the attention blocks.
        token_type_ids: (batch, seq) - only used for NSP, passed to token type embedding.
        '''
        pos = t.arange(x.shape[1], device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(pos) + self.token_type_embedding(token_type_ids if token_type_ids is not None else t.zeros_like(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, make_additive_attention_mask(one_zero_attention_mask)) if one_zero_attention_mask is not None else block(x) # self.training??
        return x

class BertLanguageModel(nn.Module):
    def __init__(self, config: transformer_replication.TransformerConfig):
        super().__init__()
        self.bert_common = BERTCommon(config)      
        self.token_embedding_bias = BiasLayer(config.vocab_size)
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.gelu = nn.GELU()
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)  

    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None, token_type_ids: Optional[t.Tensor] = None):
        x = self.bert_common(x, one_zero_attention_mask, token_type_ids)
        x = self.linear(x)
        x = self.gelu(x)
        x = self.layer_norm(x)
        x = einsum('num_embeddings embedding_dim,batch seq_len embedding_dim ->batch seq_len num_embeddings', self.bert_common.token_embedding.weight, x)
        x = self.token_embedding_bias(x)
        return x

def make_additive_attention_mask(one_zero_attention_mask: t.Tensor, big_negative_number: float = -10000) -> t.Tensor:
    '''
    one_zero_attention_mask: 
        shape (batch, seq)
        Contains 1 if this is a valid token and 0 if it is a padding token.

    big_negative_number:
        Any negative number large enough in magnitude that exp(big_negative_number) is 0.0 for the floating point precision used.

    Out: 
        shape (batch, nheads=1, seqQ=1, seqK)
        Contains 0 if attention is allowed, big_negative_number if not.
    '''
    expanded_one_zero_attention_mask = repeat(one_zero_attention_mask, 'batch seq -> batch nheads seqQ seq', nheads=1, seqQ=1)
    return  t.where(expanded_one_zero_attention_mask.bool(), 0, big_negative_number)

if __name__ == "__main__":
    arena_utils.test_make_additive_attention_mask(make_additive_attention_mask)


class BERTIMDBClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_common = BERTCommon(config)
        self.token_embedding_bias = BiasLayer(config.vocab_size) # Need to leave this here to import previous models
        self.dropout = nn.Dropout(config.dropout)
        self.linear_sentiment = nn.Linear(config.hidden_size, 2)
        self.linear_stars = nn.Linear(config.hidden_size, 1)

    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None, token_type_ids: Optional[t.Tensor] = None):
        x = self.bert_common(x, one_zero_attention_mask, token_type_ids)
        x = x[:, 0, :]
        x = self.dropout(x)
        sentiment = self.linear_sentiment(x)
        stars = 5 * self.linear_stars(x) + 5
        return sentiment, stars

#%%
bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

config = transformer_replication.TransformerConfig(
        num_layers=12,
        num_heads=12,
        vocab_size=28996,
        hidden_size=768,
        max_seq_len=512,
        dropout=0.1,
        layer_norm_epsilon=1e-12
    )

my_bert = BertLanguageModel(config)
my_bert = gpt2_replication.copy_weights(my_bert, bert, gpt2=False)
#arena_utils.print_param_count(my_bert, bert, use_state_dict=False)

def predict(model, tokenizer, text: str, k=15) -> List[List[str]]:
    '''
    Return a list of k strings for each [MASK] in the input.
    '''
    model.eval()
    device = next(model.parameters()).device

    input_ids = t.tensor(tokenizer.encode(text), dtype=t.int64, device=device).unsqueeze(0)

    prediction = []
    with t.inference_mode():
        output = model(input_ids)
        all_logits = output if isinstance(output, t.Tensor) else output.logits
        for i, input_id in enumerate(input_ids[0]):
            if input_id != 103:
                continue
            logits = all_logits[0, i, :]
            values, indices = t.topk(logits, k)
            prediction.append(tokenizer.decode(indices))
    return prediction
        
def test_bert_prediction(predict, model, tokenizer):
    '''Your Bert should know some names of American presidents.'''
    text = "Former President of the United States of America, George[MASK][MASK]"
    predictions = predict(model, tokenizer, text)
    print(f"Prompt: {text}")
    print("Model predicted: \n", "\n".join(map(str, predictions)))
    assert "Washington" in predictions[0]
    assert "Bush" in predictions[0]

if __name__ == "__main__":
    test_bert_prediction(predict, bert, tokenizer)
    test_bert_prediction(predict, my_bert, tokenizer)

# %%
