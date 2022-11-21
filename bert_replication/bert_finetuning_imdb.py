#%%
import importlib
import os
import random
import tarfile
import time
from dataclasses import dataclass
from typing import List, Union, Optional

import pandas as pd
import torch as t
import transformers
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
from tqdm.notebook import tqdm_notebook

import bert_replication
import gpt2_replication
import transformer_replication
import utils
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"
#%%
IMDB_URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
DATA_FOLDER = "./data/bert-imdb/"
IMDB_PATH = os.path.join(DATA_FOLDER, "acllmdb_v1.tar.gz")
SAVED_TOKENS_PATH = os.path.join(DATA_FOLDER, "tokens.pt")

device = t.device("cuda" if t.cuda.is_available() else "cpu")

os.makedirs(DATA_FOLDER, exist_ok=True)
utils.maybe_download(IMDB_URL, IMDB_PATH)
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
            _, split, sentiment, _, stars, _ = s
            if sentiment == "unsup":
                continue
            reviews.append(Review(split, sentiment == "pos", int(stars), str(file.read().decode('utf-8'))))
    return reviews

reviews = load_reviews(IMDB_PATH)
assert sum((r.split == "train" for r in reviews)) == 25000
assert sum((r.split == "test" for r in reviews)) == 25000

#%%
df = pd.DataFrame(reviews)
#px.histogram(df['text'].apply(lambda x: len(x))).show()
#px.histogram(df[df['split'] == 'test']['is_positive']).show()
#px.histogram(df['stars']).show()
# TODO check lingua, ftfy
# TODO better truncation, data cleaning
# %%
train_samples = 5500
test_samples = 250

def to_dataset(tokenizer: Union[transformers.PreTrainedTokenizer, transformers.PreTrainedTokenizerFast], reviews: List[Review], size: int) -> TensorDataset:
    '''Tokenize the reviews (which should all belong to the same split) and bundle into a TensorDataset.

    The tensors in the TensorDataset should be (in this exact order):

    input_ids: shape (batch, sequence length), dtype int64
    attention_mask: shape (batch, sequence_length), dtype int
    sentiment_labels: shape (batch, ), dtype int
    star_labels: shape (batch, ), dtype int
    '''
    random.shuffle(reviews)
    df = pd.DataFrame(reviews[:size])
    batch_encoding = tokenizer(df['text'].values.tolist(), padding=True, max_length=512, truncation=True, return_tensors='pt')
    #print(reviews[999])
    sentiment_labels = t.Tensor([1 if review.is_positive else 0 for review in tqdm(reviews[:size])])
    #print(sentiment_labels)
    star_labels = t.Tensor([review.stars for review in tqdm(reviews[:size])])

    return TensorDataset(batch_encoding.input_ids, batch_encoding.attention_mask, sentiment_labels, star_labels)

tokenizer = transformers.AutoTokenizer.from_pretrained("bert-base-cased")
train_data = to_dataset(tokenizer, [r for r in reviews if r.split == "train"], train_samples)
test_data = to_dataset(tokenizer, [r for r in reviews if r.split == "test"], test_samples)
t.save((train_data, test_data), SAVED_TOKENS_PATH)
# %%

class BERTIMDBClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert_common = bert_replication.BERTCommon(config)
        self.token_embedding_bias = bert_replication.BiasLayer(config.vocab_size) # Need to leave this here to import previous models
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
def train():

    wandb.init()

    config = transformer_replication.TransformerConfig(
            num_layers=12,
            num_heads=12,
            vocab_size=28996,
            hidden_size=768,
            max_seq_len=512,
            dropout=0.1,
            layer_norm_epsilon=1e-12
        )

    bert = transformers.BertForMaskedLM.from_pretrained("bert-base-cased")
    my_bert = bert_replication.BertLanguageModel(config)
    my_bert = gpt2_replication.copy_weights(my_bert, bert, gpt2=False)
    my_bert_classifier = BERTIMDBClassifier(config)
    my_bert_classifier.bert_common = gpt2_replication.copy_weights(my_bert_classifier.bert_common, my_bert.bert_common, gpt2=False)
    #print(my_bert_classifier.bert_common.positional_embedding.weight[0][0], my_bert.bert_common.positional_embedding.weight[0][0])
    #assert t.testing.assert_close(my_bert_classifier.bert_common.token_embedding.weight[0][0], my_bert.bert_common.token_embedding.weight[0][0])

    epochs = wandb.config.epochs
    batch_size = wandb.config.batch_size
    lr = wandb.config.lr
    w = wandb.config.w

    verbose=False

    model = my_bert_classifier.to(device).train()
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sentiment_loss_fn = nn.CrossEntropyLoss()
    star_loss_fn = nn.functional.l1_loss

    examples_seen = 0
    start_time = time.time()

    trainloader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    for epoch in range(epochs):
        progress_bar = tqdm_notebook(trainloader)

        for (input_ids, attention_mask, sentiment_labels, star_labels) in progress_bar:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            sentiment_labels = sentiment_labels.to(device)
            star_labels = star_labels.to(device)
            optimizer.zero_grad()
            sentiment_pred, star_pred = model(input_ids, attention_mask)

            sentiment_loss = sentiment_loss_fn(sentiment_pred, sentiment_labels.long())
            star_loss = star_loss_fn(star_pred.squeeze(), star_labels)
            loss = (1 - w) * sentiment_loss + w * star_loss
            if verbose and examples_seen % 100 == 0:
                # print(tokenizer.decode(input_ids[0]))
                # print(attention_mask)
                print("sentiment", sentiment_pred, sentiment_labels, sentiment_loss)
                print("star", star_pred.squeeze(), star_labels, star_loss)
                print()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            #print(f"Epoch = {epoch}, Loss = {loss.item():.4f}", f"Training accuracy {(sentiment_pred.argmax(-1) == sentiment_labels).sum().item()/batch_size:.0%}")
            wandb.log({"train_loss": loss, "sentiment_loss": sentiment_loss, "star_loss": star_loss, "elapsed": time.time() - start_time, "train_sentiment_accuracy": (sentiment_pred.argmax(-1) == sentiment_labels).sum().item()/batch_size, "train_star_accuracy": (star_pred.argmax(-1) == star_labels).sum().item()/batch_size}, step=examples_seen)

            examples_seen += len(input_ids)

        with t.inference_mode():
            sentiment_accuracy = 0
            star_accuracy = 0
            total = 0
            for (input_ids, attention_mask, sentiment_labels, star_labels) in tqdm_notebook(testloader):
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                sentiment_labels = sentiment_labels.to(device)
                star_labels = star_labels.to(device)
                sentiment_pred, star_pred = model(input_ids, attention_mask)
                if total == 0:
                    print(sentiment_pred, sentiment_labels)
                    print(t.round(star_pred.squeeze()), star_labels)
                sentiment_accuracy += (sentiment_pred.argmax(-1) == sentiment_labels).sum().item()
                star_accuracy += (t.round(star_pred.squeeze()) == star_labels).sum().item()
                total += sentiment_pred.size(0)
            
            wandb.log({"test_sentiment_accuracy": sentiment_accuracy/total, "test_star_accuracy": star_accuracy/total}, step=examples_seen)

        filename = f"{wandb.run.dir}/model_state_dict.pt"
        print(f"Saving model to: {filename}")
        t.save(model.state_dict(), filename)
        wandb.save(filename)

#DONE: why are all labels 0? -> Need to shuffle reviews between pos/neg
#DONE: Check a batch from the dataloader to see if data looks ok -> yes
#DONE: Classification loss starts at log(2)
#DONE: why are predictions the opposite way? -> do I need attention mask in test?


# %%
device = t.device('cpu')
sweep_config = {
        'method': 'bayes',
        'name': 'meg_bert_finetune',
        'metric': {'name': 'train_loss', 'goal': 'minimize'},
        'parameters': 
        {
            'batch_size': {'values': [16]},
            'lr': {'values': [1e-5]}, #'lr': {'max': 6e-5, 'min': 1e-5, 'distribution': 'log_uniform_values'}
            'weight_decay': {'values': [0.01]},
            'epochs': {'values': [1]},
            'w': {'values': [0.02]}
        }
    }
sweep_id = wandb.sweep(sweep=sweep_config, project='meg_bert_finetune')
wandb.agent(sweep_id=sweep_id, function=train, count=1)
#%% Test sentiment
config = transformer_replication.TransformerConfig(
            num_layers=12,
            num_heads=12,
            vocab_size=28996,
            hidden_size=768,
            max_seq_len=512,
            dropout=0.1,
            layer_norm_epsilon=1e-12
        )
model = utils.load_transformer('rkb6zszq', BERTIMDBClassifier, config) #1jr05uo4
total = 20

count = 0
for review in random.sample(reviews, total):
    encoding = tokenizer(review.text, padding=True, max_length=512, truncation=True, return_tensors='pt')
    sentiment, stars = model(encoding.input_ids, encoding.attention_mask)
    actual = "pos" if review.is_positive else "neg"
    predicted = "pos" if sentiment.argmax(-1).item() == 1 else "neg"
    print("actual", actual, "-> predicted", predicted, "✅" if actual == predicted else "❌")
    if actual == predicted:
        count += 1
print(f"Accuracy = {(count / total):.0%}")
#%% Test stars
config = transformer_replication.TransformerConfig(
            num_layers=12,
            num_heads=12,
            vocab_size=28996,
            hidden_size=768,
            max_seq_len=512,
            dropout=0.1,
            layer_norm_epsilon=1e-12
        )
model = utils.load_transformer('rkb6zszq', BERTIMDBClassifier, config) #92gukpcg
total = 20
tolerance = 2

count = 0
for review in random.sample(reviews, total):
    encoding = tokenizer(review.text, padding=True, max_length=512, truncation=True, return_tensors='pt')
    sentiment, stars = model(encoding.input_ids, encoding.attention_mask)
    actual = review.stars
    predicted = round(stars.item(), 1)
    print("actual", actual, "-> predicted", predicted, "✅" if abs(actual-predicted) <= tolerance else "❌")
    if abs(actual-predicted) <= tolerance:
        count += 1
print(f"Accuracy = {(count / total):.0%} for tolerance {tolerance}")

# %%
