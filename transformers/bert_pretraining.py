# %%
import hashlib
import os
import sys
import zipfile

import torch as t
import transformers
from einops import rearrange
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm.notebook import tqdm_notebook
import time
from typing import List, Tuple
import wandb

import transformers_utils
import transformer_replication
import bert_replication
import matplotlib.pyplot as plt

#%%
MAIN = __name__ == "__main__"
DATA_FOLDER = "../data"
DATASET = "103"
BASE_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/"
DATASETS = {"103": "wikitext-103-raw-v1.zip", "2": "wikitext-2-raw-v1.zip"}
TOKENS_FILENAME = os.path.join(DATA_FOLDER, f"wikitext_tokens_{DATASET}.pt")

if not os.path.exists(DATA_FOLDER):
    os.mkdir(DATA_FOLDER)

path = os.path.join(DATA_FOLDER, DATASETS[DATASET])
utils.maybe_download(BASE_URL + DATASETS[DATASET], path)
expected_hexdigest = {"103": "0ca3512bd7a238be4a63ce7b434f8935", "2": "f407a2d53283fc4a49bcff21bc5f3770"}
with open(path, "rb") as f:
    actual_hexdigest = hashlib.md5(f.read()).hexdigest()
    assert actual_hexdigest == expected_hexdigest[DATASET]

print(f"Using dataset WikiText-{DATASET} - options are 2 and 103")

z = zipfile.ZipFile(path)

def decompress(*splits: str):
    return [z.read(f"wikitext-{DATASET}-raw/wiki.{split}.raw").decode("utf-8").splitlines() for split in splits]

train_text, val_text, test_text = decompress("train", "valid", "test")
#%%
#TODO: tokenize_1d
import functools
def concat_lists(list_of_lists):
    return functools.reduce(lambda x, y: x+y, list_of_lists)

def tokenize_1d(tokenizer, lines: List[str], max_seq: int) -> t.Tensor:
    """Tokenize text and rearrange into chunks of the maximum length.

    Return (batch, seq) and an integer dtype.
    """

    lines_tokenized = tokenizer(
        lines, 
        truncation=False, 
        add_special_tokens=False, 
        padding=False,
        return_token_type_ids=False,
        return_attention_mask=False,
    )
    input_ids = lines_tokenized["input_ids"]
    input_ids = concat_lists(input_ids)

    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids

def tokenize_1d_with_progress_bar(tokenizer, lines: List[str], max_seq: int, n_intervals: int = 200) -> t.Tensor:
    input_ids = []
    interval_len = len(lines) // (n_intervals - 1)
    slices = [slice(i*interval_len, (i+1)*interval_len) for i in range(n_intervals)]
    progress_bar = tqdm_notebook(slices)
    for slice_ in progress_bar:
        lines_tokenized = tokenizer(
            lines[slice_], 
            truncation=False, 
            add_special_tokens=False, 
            padding=False,
            return_token_type_ids=False,
            return_attention_mask=False,
        )
        input_ids.append(concat_lists(lines_tokenized["input_ids"]))

    input_ids = concat_lists(input_ids)
    n_to_truncate = len(input_ids) % max_seq
    input_ids = t.tensor(input_ids[:-n_to_truncate]).to(t.int)

    input_ids = rearrange(input_ids, "(b s) -> b s", s=max_seq)

    return input_ids

if MAIN:
    tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    if not os.path.exists(TOKENS_FILENAME):
        max_seq = 128
        print("Tokenizing training text...")
        train_data = tokenize_1d_with_progress_bar(tokenizer, train_text, max_seq)
        print("Training data shape is: ", train_data.shape)
        print("Tokenizing validation text...")
        val_data = tokenize_1d_with_progress_bar(tokenizer, val_text, max_seq)
        print("Tokenizing test text...")
        test_data = tokenize_1d_with_progress_bar(tokenizer, test_text, max_seq)
        print("Saving tokens to: ", TOKENS_FILENAME)
        t.save((train_data, val_data, test_data), TOKENS_FILENAME)
    else:
        print(f"Loading data from {TOKENS_FILENAME}")
        train_data, val_data, test_data = t.load(TOKENS_FILENAME)
# %%
def random_mask(
    input_ids: t.Tensor, mask_token_id: int, vocab_size: int, select_frac=0.15, mask_frac=0.8, random_frac=0.1
) -> Tuple[t.Tensor, t.Tensor]:
    '''Given a batch of tokens, return a copy with tokens replaced according to Section 3.1 of the paper.
    input_ids: (batch, seq)
    Return: (model_input, was_selected) where:
    model_input: (batch, seq) - a new Tensor with the replacements made, suitable for passing to the BertLanguageModel. Don't modify the original tensor!
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise
    '''
    batch_len, seq_len = input_ids.shape

    all_mask_token = t.full_like(input_ids, mask_token_id)
    all_random_token = t.randint(low=0, high=vocab_size, size=input_ids.shape).to(input_ids.device)
    all_idx = rearrange(t.randperm(batch_len * seq_len), '(b s) -> b s', b=batch_len).to(input_ids.device)

    selection_cutoff = int(select_frac * batch_len * seq_len)
    mask_cutoff = int(mask_frac * select_frac * batch_len * seq_len)
    random_cutoff = int(random_frac * select_frac * batch_len * seq_len)

    selection_mask = all_idx < selection_cutoff
    mask_mask = all_idx < mask_cutoff
    random_mask = (all_idx < mask_cutoff + random_cutoff).to(input_ids.device)

    model_input = t.where(random_mask, all_random_token, input_ids)
    model_input = t.where(mask_mask, all_mask_token, model_input)

    return model_input, selection_mask

if MAIN:
    transformers_utils.test_random_mask(random_mask, input_size=10000, max_seq=128)
# %%
def calculate_cross_entropy_unigram(data):
    freqs = t.bincount(data.flatten())
    freqs = freqs[freqs > 0]
    probs = freqs / sum(freqs)
    cross_entropy = (- probs * probs.log()).sum()
    return cross_entropy

if MAIN:
    cross_entropy = calculate_cross_entropy_unigram(train_data)
    print(cross_entropy)

def cross_entropy_selected(pred: t.Tensor, target: t.Tensor, was_selected: t.Tensor) -> t.Tensor:
    '''
    pred: (batch, seq, vocab_size) - predictions from the model
    target: (batch, seq, ) - the original (not masked) input ids
    was_selected: (batch, seq) - 1 if the token at this index will contribute to the MLM loss, 0 otherwise

    Out: the mean loss per predicted token
    '''
    was_selected_flat = rearrange(was_selected, 'b s -> (b s)').long()
    pred_flat = rearrange(pred, 'b s v -> (b s) v')[was_selected_flat == 1]
    target_flat = rearrange(target, 'b s -> (b s)')[was_selected_flat == 1]

    return F.cross_entropy(pred_flat, target_flat)

if MAIN:
    transformers_utils.test_cross_entropy_selected(cross_entropy_selected, verbose=True)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")
# %%
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

hidden_size = 512
bert_config_tiny = transformer_replication.TransformerConfig(
    num_layers = 8,
    num_heads = hidden_size // 64,
    vocab_size = 28996,
    hidden_size = hidden_size,
    max_seq_len = 128,
    dropout = 0.1,
    layer_norm_epsilon = 1e-12
)

config_dict = dict(
    lr=0.0002,
    epochs=40,
    batch_size=128,
    weight_decay=0.01,
    mask_token_id=tokenizer.mask_token_id,
    warmup_step_frac=0.01,
    eps=1e-06,
    max_grad_norm=None,
)

(train_data, val_data, test_data) = t.load("../data/wikitext_tokens_2.pt")
print("Training data size: ", train_data.shape)

train_loader = DataLoader(TensorDataset(train_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True)
val_loader = DataLoader(TensorDataset(val_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True)
test_loader = DataLoader(TensorDataset(test_data), shuffle=True, batch_size=config_dict["batch_size"], drop_last=True)

# %%
def lr_for_step(step: int, max_step: int, max_lr: float, warmup_step_frac: float):
    '''Return the learning rate for use at this step of training.'''
    start_lr = max_lr / 10
    warmup_step = int(warmup_step_frac * max_step)
    if step < warmup_step:
        return start_lr + (max_lr - start_lr) * (step / warmup_step)
    else:
        return max_lr - (max_lr - start_lr) * ((step - warmup_step) / (max_step - warmup_step))


if MAIN:
    max_step = int(len(train_loader) * config_dict["epochs"])
    lrs = [
        lr_for_step(step, max_step, max_lr=config_dict["lr"], warmup_step_frac=config_dict["warmup_step_frac"])
        for step in range(max_step)
    ]
    plt.plot(lrs)
# %%
def make_optimizer(model: bert_replication.BertLanguageModel, config_dict: dict) -> t.optim.AdamW:
    '''
    Loop over model parameters and form two parameter groups:

    - The first group includes the weights of each Linear layer and uses the weight decay in config_dict
    - The second has all other parameters and uses weight decay of 0
    '''
    linear_group = []
    other_group = []
    for name, param in model.named_parameters():
        if ".weight" in name and 'layer_norm' not in name and 'embedding' not in name:
            linear_group.append(param)
        else:
            other_group.append(param)
    return t.optim.AdamW([
        {'params': linear_group, 'weight_decay': config_dict['weight_decay']}, 
        {'params': other_group}], lr=config_dict['lr'], eps=config_dict['eps'])


if MAIN:
    test_config = transformer_replication.TransformerConfig(
        num_layers = 3,
        num_heads = 1,
        vocab_size = 28996,
        hidden_size = 1,
        max_seq_len = 4,
        dropout = 0.1,
        layer_norm_epsilon = 1e-12,
    )

    optimizer_test_model = bert_replication.BertLanguageModel(test_config)
    opt = make_optimizer(
        optimizer_test_model, 
        dict(weight_decay=0.1, lr=0.0001, eps=1e-06)
    )
    expected_num_with_weight_decay = test_config.num_layers * 6 + 1
    wd_group = opt.param_groups[0]
    actual = len(wd_group["params"])
    assert (
        actual == expected_num_with_weight_decay
    ), f"Expected 6 linear weights per layer (4 attn, 2 MLP) plus the final lm_linear weight to have weight decay ({expected_num_with_weight_decay}), got {actual}"
    all_params = set()
    for group in opt.param_groups:
        all_params.update(group["params"])
    assert all_params == set(optimizer_test_model.parameters()), "Not all parameters were passed to optimizer!"
# %%
def bert_mlm_pretrain(model: bert_replication.BertLanguageModel, config_dict: dict, train_loader: DataLoader, verbose=False) -> None:
    '''Train using masked language modelling.'''

    wandb.init(project='meg_bert_mlm_pretrain', config=config_dict)

    device = t.device('cuda' if t.cuda.is_available() else 'cpu')

    model = model.to(device).train()
    optimizer = make_optimizer(model, config_dict)
    loss_fn = cross_entropy_selected
    epochs = config_dict['epochs']

    trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    max_step = epochs * len(trainloader)

    examples_seen = 0
    step = 0
    start_time = time.time()

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10, log_graph=True)

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)

        for x in progress_bar:
            x = x.to(device)
            model_input, was_selected = random_mask(x, config_dict['mask_token_id'], tokenizer.vocab_size)
            optimizer.zero_grad()
            x_pred = model(model_input)
            loss = loss_fn(x_pred, x.long(), was_selected)
            if verbose and examples_seen % 100 == 0:
                #print(x.shape, x_pred.shape)
                #print(len(optimizer.param_groups))
                pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) # small model and we want to solve a problem perfectly
            current_lr = lr_for_step(step, max_step, config_dict['lr'], config_dict['warmup_step_frac'])
            for group in optimizer.param_groups:
                group['lr'] = current_lr
            optimizer.step()
            train_accuracy = (x_pred.argmax(-1) == x).sum().item() / was_selected.sum() / batch_size
            #print(f"Epoch = {epoch}, Loss = {loss.item():.4f}, Training accuracy {train_accuracy:.0%}")
            wandb.log({"train_loss": loss, "train_accuracy": train_accuracy, "lr": current_lr, "elapsed": time.time() - start_time}, step=examples_seen)
            step += 1
            examples_seen += len(x)

        with t.inference_mode():
            test_accuracy = 0
            total = 0
            for x in tqdm_notebook(testloader):
                x = x.to(device)
                model_input, was_selected = random_mask(x, config_dict['mask_token_id'], tokenizer.vocab_size)
                x_pred = model(model_input)
                if total == 0 and verbose:
                    pass
                test_accuracy += (x_pred.argmax(-1) == x).sum().item() / was_selected.sum()
                total += x_pred.size(0)
            
            print(test_accuracy/total)
            wandb.log({"test_accuracy": test_accuracy/total}, step=examples_seen)

        filename = f"{wandb.run.dir}/model_state_dict.pt"
        print(f"Saving model to: {filename}")
        t.save(model.state_dict(), filename)
        wandb.save(filename)

if MAIN:
    model = bert_replication.BertLanguageModel(bert_config_tiny)
    num_params = sum((p.nelement() for p in model.parameters()))
    print("Number of model parameters: ", num_params)
    bert_mlm_pretrain(model, config_dict, train_loader, verbose=True)
# %%
if MAIN:
    model = bert_replication.BertLanguageModel(bert_config_tiny)
    model.load_state_dict(t.load(f"{wandb.run.dir}/model_state_dict.pt"))
    your_text = "The Answer to the Ultimate Question of Life, The Universe, and Everything is [MASK]."
    predictions = bert_replication.predict(model, tokenizer, your_text)
    print("Model predicted: \n", "\n".join(map(str, predictions)))
# %%
