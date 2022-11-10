# %%
import torch as t
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import transformer_replication
from typing import Optional,Union
import re
import requests
import torch.nn as nn
from typing import Union, Optional
from einops import rearrange
from tqdm.notebook import tqdm_notebook
import time
import wandb
import sampling
# %%
class WordsDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        text = self.texts[idx]
        sample = (text, label)
        return sample
# %%
def tokenize(text):
    return re.split(r"\b", text)

def remove_duplicates(text, string=" "):
    if string + string in text:
        text = text.replace(string + string, string)
        return remove_duplicates(text, string)
    return text

# %%
class Data():
    def __init__(self, text, start=None, end=None):
        self.complete_text = remove_duplicates(remove_duplicates(text, " "), "\n")
        if start is not None and end is not None:
            self.complete_text = self.get_excerpt(start, end)
        self.complete_tokens = tokenize(self.complete_text)
        self.vocab = sorted(set(self.complete_tokens))
        self.token_to_id = dict(zip(self.vocab, list(range(len(self.vocab)))))
        self.id_to_token = dict(zip(list(range(len(self.vocab))), self.vocab))
        self.model_max_length = None

    @staticmethod
    def from_link(link, start=None, end=None):
        return Data(requests.get(link).content.decode('utf-8'), start, end)

    @staticmethod
    def from_file(filename, start=None, end=None):
        with open(filename, encoding='utf-8') as f:
            text = f.read()
        return Data(text, start, end)

    def get_excerpt(self, start, end, text=None):
        if text is None:
            text = self.complete_text
        return text.split(start, maxsplit=1)[1].split(end, maxsplit=1)[0]

    def generate_autoregressive_dataset(self, sequence_length, text=None):
        self.model_max_length = sequence_length
        if text is None:
            text = self.complete_text
        token_ids = self.encode(text, return_tensors="pt")
        inputs = [token_ids[i:i + sequence_length] for i in range(len(token_ids) - sequence_length)]
        labels = [token_ids[i + 1:i + 1 + sequence_length] for i in range(len(token_ids) - sequence_length)]
        return WordsDataset(inputs, labels)

    def encode(self, initial_text: str, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        tokens = tokenize(initial_text)
        token_ids = [self.token_to_id[t] for t in tokens]
        if return_tensors == "pt":
            return t.tensor(token_ids)
        return token_ids

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        tokens = [self.id_to_token[int(i)] for i in list_of_ids]
        return "".join(tokens)

shakespeare = Data.from_file('data/shakespeare.txt', start="From ", end='END')
#%%
import wandb
device = t.device('cpu')

config = None

def train():
    #wandb.init()
    
    wandb_config_dict = {
        'batch_size': 64,
        'hidden_size': 256,
        'lr': 0.001,
        'max_seq_len': 40
    }
    wandb.init(project='shakespeare', config=wandb_config_dict)

    config = transformer_replication.TransformerConfig(
        num_layers=6, #N=6
        num_heads=8, #h=8
        vocab_size=len(shakespeare.vocab),
        hidden_size=wandb.config.hidden_size, #d_model = 64 x 8 = 512
        max_seq_len=wandb.config.max_seq_len,
        dropout=0.1 #p=0.1
    )

    epochs = 2
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    model = transformer_replication.DecoderOnlyTransformer(config).to(device).train()
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    examples_seen = 0
    start_time = time.time()

    traintext = shakespeare.get_excerpt("fairest", "150")
    testtext = shakespeare.get_excerpt("150", "THE ")

    trainset = shakespeare.generate_autoregressive_dataset(config.max_seq_len, traintext)
    testset = shakespeare.generate_autoregressive_dataset(config.max_seq_len, testtext)

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
    shakespeare.model_max_length = wandb.config.max_seq_len
    text_output = sampling.sample_tokens(model, shakespeare, ' It was a beautiful ', max_tokens_generated=100, temperature=1.0, top_k=10)
    print(text_output)
    return model

"""
sweep_config = {
        'method': 'bayes',
        'name': 'shakespeare_sweep',
        'metric': {'name': 'test_accuracy', 'goal': 'maximize'},
        'parameters': 
        {
            'batch_size': {'values': [64]},
            'hidden_size': {'values': [64]},
            'max_seq_len': {'values': [20]},
            'lr': {'values': [0.001]},
            #'lr': {'max': 0.05, 'min': 0.0001, 'distribution': 'log_uniform_values'}
        }
    }

sweep_id = wandb.sweep(sweep=sweep_config, project='shakespeare')

wandb.agent(sweep_id=sweep_id, function=train, count=1)
"""
model = train()
# %%

#print(wandb.run.dir)
#model = transformer_replication.DecoderOnlyTransformer(config)
#model.load_weights(wandb.restore("/Users/m/Documents/arena/wandb/run-20221110_103122-1qjib12l/files/model_state_dict.pt"))
shakespeare.model_max_length = wandb.config.max_seq_len
text_output = sampling.sample_tokens(model, shakespeare, ' from fairest ', max_tokens_generated=100, temperature=1.0, top_k=10)
print(text_output)
# %%
