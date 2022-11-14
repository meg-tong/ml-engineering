#%%
import os
import re
import tarfile
from dataclasses import dataclass
import requests
import torch as t
import transformers
from torch.utils.data import TensorDataset
from tqdm.auto import tqdm
import plotly.express as px
import pandas as pd
from typing import Callable, Optional, List, Union
import time
import transformer_replication
import bert_replication
#%%
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")

device = t.device("cuda" if t.cuda.is_available() else "cpu")

def maybe_download(url: str, path: str) -> None:
    '''
    Download the file from url and save it to path. 
    If path already exists, do nothing.
    '''
    if os.path.isfile(path):
        return
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.raw.read())

os.makedirs(DATA_FOLDER, exist_ok=True)
maybe_download(IMDB_URL, IMDB_PATH)
#%%
@dataclass(frozen=True)
class Review:
    split: str          # "train" or "test"
    is_positive: bool   # sentiment classification
    stars: int          # num stars classification
    text: str           # text content of review

def load_reviews(path: str) -> List[Review]:
    files = tarfile.open(path)
    reviews = []
    for member in tqdm(files.getmembers()):
        file = files.extractfile(member)
        s = member.name.replace("_", "/").replace(".", "/").split('/')
        if file is not None and len(s) == 6:
            _, split, is_positive, _, stars, _ = s
            if is_positive == "unsup":
                continue
            reviews.append(Review(split, bool(is_positive), int(stars), str(file.read().decode('utf-8'))))
    return reviews

reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000
# %%
df = pd.DataFrame(reviews)
px.histogram(df['text'].apply(lambda x: len(x))).show()
px.histogram(df['stars']).show()
# %%
# TODO check lingua, ftfy
# TODO better truncation, data cleaning
# %%
MAX_TOKENS = 512
def to_dataset(tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast], reviews: List[Review]) -> TensorDataset:
    '''Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    '''
    df = pd.DataFrame(reviews)
    batch_encoding = tokenizer(df['text'].values.tolist(), padding=True, max_length=512, truncation=True, return_tensors='pt')
    sentiment_labels = t.Tensor([1 if review.is_positive else 0 for review in tqdm(reviews)])
    star_labels = t.Tensor([review.stars for review in tqdm(reviews)])

    return TensorDataset(batch_encoding.input_ids, batch_encoding.attention_mask, sentiment_labels, star_labels)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"])
test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"])
t.save((train_data, test_data), SAVED_TOKENS_PATH)
# %%
config = transformer_replication.TransformerConfig(
        num_layers=12,
        num_heads=12,
        vocab_size=28996,
        hidden_size=768,
        max_seq_len=512,
        dropout=0.1,
        layer_norm_epsilon=1e-12
    )

my_bert_classifier = bert_replication.BERTClassifier(config)

 # %%
