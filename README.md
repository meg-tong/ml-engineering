This is my work from a research engineering accelerator program (ARENA). The program follows a combination of Redwood Research's MLAB and Jacob Hilton's [Deep Learning Curriculum](https://github.com/jacobhilton/deep_learning_curriculum).

#### Writing implementations
- Implementing important components of PyTorch from scratch, including the backprop algorithm, modules (e.g. `Conv2d`, `BatchNorm2d`), optimizers (e.g. `Adam`) and learning rate schedulers
- Implementing transformers from scratch, including the attention mechanism and sampling algorithms, then using my implementation to replicate GPT-2 and BERT
- Implementing reinforcement learning algorithms, including DQN, PPO, SARSA, Q-learning and a few multi-armed bandit algos
- Implementing models with other objectives, including DCGANs and VAEs

#### Running experiments
- Training my own small transformer architecture on a LeetCode problem: `transformers/leetcode_transformer.py`
- Training a GPT-2-like transformer on the Shakespeare corpus: `transformers/shakespeare_transformer.py`
- Pre-training BERT on WikiText: `transformers/bert_pretraining.py`
- Fine-tuning BERT to predict movie reviews: `transformers/bert_finetuning_imdb.py`
- Training a DQN agent on CartPole: `reinforcement_learning/dqn.py`
