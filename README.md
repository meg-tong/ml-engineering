This is my work from the Alignment Research Engineer Accelerator program (ARENA), which follows Jacob Hilton's [Deep Learning Curriculum](https://github.com/jacobhilton/deep_learning_curriculum).

#### Writing implementations
- Implementing important components of PyTorch from scratch, including the backprop algorithm, modules (e.g. `Conv2d`, `BatchNorm2d`), optimizers (e.g. `Adam`) and learning rate schedulers
- Implementing transformers from scratch, including the attention mechanism and sampling algorithms, then using my implementation to replicate GPT-2 and BERT

#### Running experiments
- Training my own small transformer architecture on a LeetCode problem: `leetcode_transformer.py`
- Training a GPT-2-like transformer on the Shakespeare corpus: `shakespeare_transformer.py`
- Pre-training BERT on WikiText: `bert_pretraining.py`
- Fine-tuning BERT to predict movie reviews: `bert_finetuning_imdb.py`
