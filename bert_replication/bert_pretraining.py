# %%
import hashlib
import os
import sys
import zipfile

import torch as t
import transformers
from einops import rearrange, repeat
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
from typing import List, Tuple
import collections
import numpy as np

sys.path.append("../")
import importlib
importlib.reload(arena_utils)
import arena_utils
import utils

#%%
MAIN = __name__ == "__main__"
DATA_FOLDER = "../data"
DATASET = "2"
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
tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")

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

if MAIN:
    max_seq = 128
    print("Tokenizing training text...")
    train_data = tokenize_1d(tokenizer, train_text, max_seq)
    print("Training data shape is: ", train_data.shape)
    print("Tokenizing validation text...")
    val_data = tokenize_1d(tokenizer, val_text, max_seq)
    print("Tokenizing test text...")
    test_data = tokenize_1d(tokenizer, test_text, max_seq)
    print("Saving tokens to: ", TOKENS_FILENAME)
    t.save((train_data, val_data, test_data), TOKENS_FILENAME)
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
    all_random_token = t.randint(low=0, high=vocab_size, size=input_ids.shape)

    all_idx = rearrange(t.randperm(batch_len * seq_len), '(b s) -> b s', b=batch_len)
    selection_cutoff = int(select_frac * batch_len * seq_len)
    mask_cutoff = int(mask_frac * select_frac * batch_len * seq_len)
    random_cutoff = int(random_frac * select_frac * batch_len * seq_len)
    selection_mask = all_idx < selection_cutoff
    mask_mask = all_idx < mask_cutoff
    random_mask = all_idx < mask_cutoff + random_cutoff

    model_input = t.where(random_mask, all_random_token, input_ids)
    model_input = t.where(mask_mask, all_mask_token, model_input)

    return model_input, selection_mask

if MAIN:
    arena_utils.test_random_mask(random_mask, input_size=10000, max_seq=max_seq)
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
# %%
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
    arena_utils.test_cross_entropy_selected(cross_entropy_selected, verbose=True)

    batch_size = 8
    seq_length = 512
    batch = t.randint(0, tokenizer.vocab_size, (batch_size, seq_length))
    pred = t.rand((batch_size, seq_length, tokenizer.vocab_size))
    (masked, was_selected) = random_mask(batch, tokenizer.mask_token_id, tokenizer.vocab_size)
    loss = cross_entropy_selected(pred, batch, was_selected).item()
    print(f"Random MLM loss on random tokens - does this make sense? {loss:.2f}")