#%%
import os
import random
import time
from typing import Optional, Union

import numpy as np
import torch as t
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.notebook import tqdm_notebook
import wandb

import bert_replication
import transformer_replication
import utils

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%
class LeetCodeTokenizer():
    def __init__(self):
        self.token_to_id = {'[CLS]': 0, '[SEP]': 1, '[PAD]': 2, '(': 3, ')': 4}
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}

    def encode(self, initial_text: str, length: int = 8, return_tensors: Optional[str] = None) -> Union[list, t.Tensor]:
        '''
        Tokenizes initial_text, then returns the token ids.

        Return type is list by default, but if return_tensors="pt" then it is returned as a tensor.
        '''
        padding = length - len(initial_text) - 2
        token_ids = [0] + [self.token_to_id[char] for char in initial_text] + [1] + [2] * padding
        if return_tensors == "pt":
            return t.tensor(token_ids)
        return token_ids

    def decode(self, list_of_ids: Union[t.Tensor, list]) -> str:
        '''
        Converts ids to a list of tokens, then joins them into a single string.
        '''
        tokens = [self.id_to_token[int(i)] for i in list_of_ids]
        return "".join(tokens)

class LeetCode():
    def __init__(self):
        self.bracket_to_number = {'(': 1, ')': -1}
        self.number_to_bracket = {v: k for k, v in self.bracket_to_number.items()}

    def convert_brackets_to_numbers(self, brackets):
        return [self.bracket_to_number[b] for b in brackets]

    def convert_numbers_to_brackets(self, numbers):
        return [self.number_to_bracket[n] for n in numbers]

    def is_balanced(self, numbers):
        if isinstance(numbers[0], str):
            return self.is_balanced(self.convert_brackets_to_numbers(numbers))
        elif isinstance(numbers, str):
            return self.is_balanced(numbers.split())
        altitude = 0
        for n in numbers:
            altitude += n
            if altitude < 0:
                return False
        return altitude == 0

    def generate_true_string(self, length=6):
        numbers = [1] * round(length / 2) + [-1] * round(length / 2)
        random.shuffle(numbers)
        altitude = 0
        fixed_numbers = numbers.copy()
        for i, n in enumerate(numbers):
            if altitude + n < 0 or altitude < 0:
                fixed_numbers[i] = -fixed_numbers[i]
            altitude += n
        return ''.join(self.convert_numbers_to_brackets(fixed_numbers))

    #TODO: try context-free language S -> SS | (S) | epsilon

    def generate_false_string(self, length=6):
        numbers = [1] * round(length / 2) + [-1] * round(length / 2)
        random.shuffle(numbers)
        if self.is_balanced(numbers):
            return self.generate_false_string(length)
        return ''.join(self.convert_numbers_to_brackets(numbers))

    def generate_all_strings(self, length=6):
        return [format(n, str(length) + 'b').replace(" ", "0").replace("0", "(").replace("1", ")") for n in tqdm_notebook(range(2 ** length))]
            
class LeetCodeDataset(Dataset):
    def __init__(self, strings, labels):
        self.strings = strings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        string = self.strings[idx]
        label = self.labels[idx]
        sample = (string, label)
        return sample

    @staticmethod
    def generate_dataset(leetcode: LeetCode, tokenizer: LeetCodeTokenizer, length=16, num_samples=10000):
        true_strings = [leetcode.generate_true_string(length - 2) for _ in range(2 ** length)]
        true_strings = list(set(true_strings))
        false_strings = [leetcode.generate_false_string(length - 2) for _ in range(num_samples - len(true_strings))]
        strings = true_strings + false_strings
        random.shuffle(strings)
        labels = [1 if leetcode.is_balanced(string) else 0 for string in strings]
        return LeetCodeDataset(t.Tensor([tokenizer.encode(string, length) for string in strings]), labels)
    
    @staticmethod
    def generate_smart_dataset(leetcode: LeetCode, tokenizer: LeetCodeTokenizer, length=16, true_split=0.09, test_split=0.01):
        length -= 2
        all_strings = leetcode.generate_all_strings(length)
        true_strings = []
        false_strings = []
        for string in all_strings:
            if leetcode.is_balanced(string):
                true_strings.append(string)
            else:
                false_strings.append(string)
        print(f"{len(all_strings)} total strings, with {len(false_strings)} false strings and {len(true_strings)} true strings")
        random.shuffle(true_strings)
        random.shuffle(false_strings)
        false_strings = false_strings[:int(len(true_strings) / true_split)]
        num_true_test = int(np.floor(len(true_strings) * test_split))
        num_false_test = int(np.floor(len(false_strings) * test_split))

        train_strings = true_strings[:-num_true_test] + false_strings[:-num_false_test]
        train_labels = [1 if leetcode.is_balanced(string) else 0 for string in train_strings]
        test_strings = true_strings[-num_true_test:] + false_strings[-num_false_test:]
        test_labels = [1 if leetcode.is_balanced(string) else 0 for string in test_strings]
        print(f"{len(train_strings)} train samples, with {sum(train_labels) / len(train_labels):.0%} true train samples")
        print(f"{len(test_strings)} test samples, with {sum(test_labels) / len(test_labels):.0%} true test samples")

        train_dataset = LeetCodeDataset(t.Tensor([tokenizer.encode(string, length) for string in train_strings]), train_labels)
        test_dataset = LeetCodeDataset(t.Tensor([tokenizer.encode(string, length) for string in test_strings]), test_labels)
        return train_dataset, test_dataset
        
leetcode = LeetCode()
tokenizer = LeetCodeTokenizer()
LeetCodeDataset.generate_smart_dataset(leetcode, tokenizer, length=16)
#%%
dataset = LeetCodeDataset.generate_dataset(leetcode, tokenizer)
#assert tokenizer.encode(dataset[0][0])[0] == 0
#assert tokenizer.encode(dataset[0][0])[-1] == 1
#%%


#%%
class LeetCodeTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = transformer_replication.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = transformer_replication.Embedding(config.max_seq_len, config.hidden_size)
        self.token_type_embedding = transformer_replication.Embedding(2, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*[bert_replication.BERTBlock(config) for _ in range(config.num_layers)])
        self.dropout = nn.Dropout(config.dropout)
        self.linear = nn.Linear(config.hidden_size, 2)

    def forward(self, x: t.Tensor, one_zero_attention_mask: Optional[t.Tensor] = None, token_type_ids: Optional[t.Tensor] = None):
        pos = t.arange(x.shape[1], device=x.device)
        x = self.token_embedding(x) + self.positional_embedding(pos) + self.token_type_embedding(token_type_ids if token_type_ids is not None else t.zeros_like(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x, bert_replication.make_additive_attention_mask(one_zero_attention_mask)) if one_zero_attention_mask is not None else block(x) # self.training??
        x = x[:, 0, :]
        x = self.dropout(x)
        x = self.linear(x)
        return x
#%%
def train():

    wandb.init(entity='meg-tong')

    config = transformer_replication.TransformerConfig(
            num_layers=wandb.config.num_layers,
            num_heads=wandb.config.num_heads,
            vocab_size=5,
            hidden_size=wandb.config.hidden_size,
            max_seq_len=wandb.config.max_seq_len,
            dropout=wandb.config.dropout,
            layer_norm_epsilon=1e-12
        )

    my_bert_classifier = LeetCodeTransformer(config)

    device = t.device('cpu')
    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr

    verbose=False

    model = my_bert_classifier.to(device).train()
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=wandb.config.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_data, test_data = LeetCodeDataset.generate_smart_dataset(leetcode, tokenizer, length=wandb.config.max_seq_len)

    if verbose:
        example_string, example_label = train_data[0]
        print(tokenizer.decode(example_string), bool(example_label))
        attention_mask = t.where(example_string == 2, 0, 1)
        print(attention_mask)

    trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    examples_seen = 0
    start_time = time.time()

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)

        for (x, y) in progress_bar:
            x = x.to(device)
            y = y.to(device)
            attention_mask = t.where(x == 2, 0, 1)
            optimizer.zero_grad()
            y_pred = model(x.long(), attention_mask)
            loss = loss_fn(y_pred, y.long())
            if verbose and examples_seen % 100 == 0:
                # print(tokenizer.decode(input_ids[0]))
                # print(attention_mask)
                # print(x[0], attention_mask[0])
                print(x, attention_mask, y_pred, y.long(), loss)
                print()
            loss.backward()
            #nn.utils.clip_grad_norm_(model.parameters(), 1.0) # small model and we want to solve a problem perfectly
            optimizer.step()
            train_accuracy = (y_pred.argmax(-1) == y).sum().item() / batch_size
            #print(f"Epoch = {epoch}, Loss = {loss.item():.4f}, Training accuracy {train_accuracy:.0%}")
            wandb.log({"train_loss": loss, "elapsed": time.time() - start_time, "train_accuracy": train_accuracy}, step=examples_seen)

            examples_seen += len(x)

        with t.inference_mode():
            test_accuracy = 0
            total = 0
            for (x, y) in tqdm_notebook(testloader):
                x = x.to(device)
                y = y.to(device)
                attention_mask = t.where(x == 2, 0, 1)
                y_pred = model(x.long(), attention_mask)
                if total == 0 and verbose:
                    print(y_pred, y)
                test_accuracy += (y_pred.argmax(-1) == y).sum().item()
                total += y_pred.size(0)
            
            print(test_accuracy/total)
            wandb.log({"test_accuracy": test_accuracy/total}, step=examples_seen)

        filename = f"{wandb.run.dir}/model_state_dict.pt"
        print(f"Saving model to: {filename}")
        t.save(model.state_dict(), filename)
        wandb.save(filename)
#%%
device = t.device('cpu')
sweep_config = {
        'method': 'bayes',
        'name': 'meg_leetcode_brackets',
        'metric': {'name': 'train_loss', 'goal': 'minimize'},
        'parameters': 
        {
            'num_layers': {'values': [2]},
            'num_heads': {'values': [8, 16]},
            'hidden_size': {'values': [128, 256, 512]},
            'batch_size': {'values': [256]},
            'lr': {'max': 1e-2, 'min': 1e-6, 'distribution': 'log_uniform_values'},
            'weight_decay': {'values': [0, 0.01]},
            'epochs': {'values': [1]},
            'max_seq_len': {'values': [24]},
            'dropout': {'values': [0.1, 0.2, 0.3]}
        }
    }
sweep_id = wandb.sweep(sweep=sweep_config, project='meg_leetcode_brackets')
wandb.agent(sweep_id=sweep_id, function=train, count=15)
# %%
model = utils.load_transformer('mk7htryf', LeetCodeTransformer, vocab_size=5)
train_data, test_data = LeetCodeDataset.generate_smart_dataset(leetcode, tokenizer, length=24)
#%%
total = 40
count = 0
for i in np.random.randint(0, high=len(test_data), size=total):
    string, label = test_data[i]
    string = string.unsqueeze(0)
    attention_mask = t.ones_like(string)
    output = model(string.long(), attention_mask)
    pred = output.argmax(-1).item()
    print("string", tokenizer.decode(string[0][1:-1]), "actual", label == 1, "-> predicted", pred == 1, "✅" if pred == label else "❌")
    if pred == label:
        count += 1
print(f"Accuracy = {(count / total):.0%}")
# %%
