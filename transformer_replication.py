#%% 
import transformers
import torch as t
import torch.nn as nn
import utils_w1d1
from typing import Union, List
from fancy_einsum import einsum
import torch as t
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from fancy_einsum import einsum
from typing import Union, Optional, Callable, Tuple
import numpy as np
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import time
import wandb
# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
if __name__ == "__main__":
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

if __name__ == "__main__":
    utils_w1d1.test_embedding(Embedding)
# %%
#TODO positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, embedding_dim: int):
        super().__init__()
        # Defining our positional encoding array, with `max_seq_len` rows
        # This is an advantage of using sinusoidal encoding: we can easily expand to sequences of greater length without adding more learned params
        angles = t.outer(t.arange(max_seq_len), 1 / 10000 ** (2 * t.arange(embedding_dim//2) / embedding_dim))
        pe = t.zeros((max_seq_len, embedding_dim))
        pe[:, ::2] = t.sin(angles)
        pe[:, 1::2] = t.cos(angles)
        # Register array as a buffer, rather than parameter (we don't want it to be updated by gradient descent)
        self.register_buffer('pe', pe)

    def forward(self, x: t.Tensor) -> t.Tensor:
        """
        x: shape (batch, seq_len, embedding_dim)
        """
        batch, seq_len, embedding_dim = x.shape
        # We slice the positional encoding, so it's the same shape as x
        # This is equivalent to just using an nn.Embedding, but having the input be t.arange(seq_len)
        return x + self.pe[:seq_len, :] # type: ignore


# %%
class LayerNorm(nn.Module):

    def __init__(self, normalized_shape: Union[int, List[int]], eps: float = 1e-05, elementwise_affine: bool = True):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = nn.Parameter(t.ones(normalized_shape))
            self.bias = nn.Parameter(t.zeros(normalized_shape))

    def forward(self, x: t.Tensor) -> t.Tensor:
        normalized_shape_dims = 1 if isinstance(self.normalized_shape, int) else len(self.normalized_shape)
        x_mean = x.mean(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True) # complement of the normalised shape
        x_var = x.var(dim=list(range(x.dim()))[-normalized_shape_dims:], keepdim=True, unbiased=False) # complement of the normalised shape
        x_scaled = (x - x_mean) / t.sqrt(x_var + self.eps)
        if self.elementwise_affine:
            return x_scaled * self.weight + self.bias
        return x_scaled

    def extra_repr(self) -> str:
        pass
    
if __name__ == "__main__":
    utils_w1d1.test_layernorm_mean_1d(LayerNorm)
    utils_w1d1.test_layernorm_mean_2d(LayerNorm)
    utils_w1d1.test_layernorm_std(LayerNorm)
    utils_w1d1.test_layernorm_exact(LayerNorm)
    utils_w1d1.test_layernorm_backward(LayerNorm)
# %%
from dataclasses import dataclass

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int
    max_seq_len: int
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05
# %%
import attention_replication

class BertMLP(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, 4 * config.hidden_size)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x

class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = attention_replication.MultiheadMaskedAttention(config.hidden_size, config.num_heads)
        self.layer_norm1 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.mlp = BertMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)

    def forward(self, x: t.Tensor) -> t.Tensor:
        y = self.attention(x)
        y = self.layer_norm1(y)
        x = x + y
        z = self.mlp(x)
        z = self.layer_norm2(z)
        x = x + z
        return x

class DecoderOnlyTransformer(nn.Module):

    def __init__(self, config: TransformerConfig, positional_embedding: Callable = PositionalEncoding, decoderBlock: Callable = DecoderBlock):
        super().__init__()
        self.token_embedding = Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = positional_embedding(config.max_seq_len, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.bert_blocks = nn.Sequential(*[decoderBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        
    def forward(self, x: t.Tensor) -> t.Tensor:
        if isinstance(self.positional_embedding, PositionalEncoding):
            x = self.token_embedding(x)
            x = self.positional_embedding(x)
        else:
            pos = t.arange(x.shape[1], device=x.device)
            x = self.token_embedding(x) + self.positional_embedding(pos)
        x = self.dropout(x)
        for block in self.bert_blocks:
            x = block(x)
        x = self.layer_norm(x)
        x = einsum('num_embeddings embedding_dim,batch seq_len embedding_dim ->batch seq_len num_embeddings', self.token_embedding.weight, x)
        return x

# %%
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class CustomTextDataset(Dataset):
    def __init__(self, texts, labels):
        self.labels = labels
        self.texts = texts

    @staticmethod
    def from_config(config, samples):
        texts = [t.randint(high=config.vocab_size, size=(config.max_seq_len,)) for _ in range(samples)]
        labels = [t.flip(text, (0,)) for text in texts]
        return CustomTextDataset(texts, labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample

# %%
import wandb
device = t.device('cpu')
def train():

    #wandb_config_dict = {
    #    'batch_size': 256,
    #    'hidden_size': 64,
    #    'lr': 0.00125
    #}
    
    wandb.init()

    config = TransformerConfig(
        num_layers=2, #N=6
        num_heads=4, #h=8
        vocab_size=10,
        hidden_size=wandb.config.hidden_size, #d_model = 64 x 8 = 512
        max_seq_len=6,
        dropout=0.0 #p=0.1
    )

    epochs = 1
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    train_samples = 500000
    test_samples = 1000

    model = DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    trainset = CustomTextDataset.from_config(config, train_samples)
    testset = CustomTextDataset.from_config(config, test_samples)

    trainloader = DataLoader(trainset, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(testset, shuffle=True, batch_size=batch_size)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = loss_fn(rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size"), rearrange(y, "batch seq -> (batch seq)"))
            loss.backward()
            optimizer.step()
            progress_bar.set_description(f"Epoch = {epoch}, Loss = {loss.item():.4f}")
            examples_seen += len(x)
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time}, step=examples_seen)

        with t.inference_mode():
            accuracy = 0
            total = 0
            for (x, y) in testloader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                y_flat = rearrange(y, "batch seq -> (batch seq)")
                y_pred_flat = rearrange(y_hat, "batch seq vocab_size -> (batch seq) vocab_size")
                y_predictions = y_pred_flat.argmax(-1)
                accuracy += (y_predictions == y_flat).sum().item()
                total += y_flat.size(0)

            wandb.log({"test_accuracy": accuracy/total}, step=examples_seen)

        print(f"Epoch {epoch+1}/{epochs}, train loss is {loss:.6f}, accuracy is {accuracy}/{total}")

    filename = f"{wandb.run.dir}/model_state_dict.pt"
    print(f"Saving model to: {filename}")
    t.save(model.state_dict(), filename)
    wandb.save(filename)
    return model

sweep_config = {
    'method': 'bayes',
    'name': 'w1d1_transformer_fixed',
    'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
    'parameters': 
    {
        'batch_size': {'values': [256]},
        'hidden_size': {'values': [64, 128, 256]},
        'lr': {'max': 0.05, 'min': 0.0001, 'distribution': 'log_uniform_values'}
     }
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_config, project='w1d1_transformer_fixed')

    wandb.agent(sweep_id=sweep_id, function=train, count=15)

# %%
if False:
    print(model(t.tensor([[1, 2, 3, 4, 5, 6]])).argmax(-1))
    print(model(t.tensor([[1, 2, 3, 4, 6, 6]])).argmax(-1))
    print(model(t.tensor([[1, 2, 3, 4, 8, 6]])).argmax(-1))
    print(model(t.tensor([[4, 4, 8, 4, 9, 6]])).argmax(-1))
# %%
