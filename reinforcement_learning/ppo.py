#%%
import argparse
import os
import random
import time
import sys
from distutils.util import strtobool
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch as t
import gym
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gym.spaces import Discrete
from typing import Any, List, Optional, Union, Tuple, Iterable
from einops import rearrange

import rl_utils

MAIN = __name__ == "__main__"
RUNNING_FROM_FILE = "ipykernel_launcher" in os.path.basename(sys.argv[0])

#%%
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    t.nn.init.orthogonal_(layer.weight, std)
    t.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    critic: nn.Sequential
    actor: nn.Sequential

    def __init__(self, envs: gym.vector.SyncVectorEnv): # multiple instances of the environment?
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(envs.single_observation_space.shape[0], 64)), # also std=sqrt(2)?
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1)
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(envs.single_observation_space.shape[0], 64)), # also std=sqrt(2)?
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
        )

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_agent(Agent)

# %%
@t.inference_mode()
def compute_advantages(
    next_value: t.Tensor,
    next_done: t.Tensor,
    rewards: t.Tensor,
    values: t.Tensor,
    dones: t.Tensor,
    device: t.device,
    gamma: float,
    gae_lambda: float,
) -> t.Tensor:
    '''Compute advantages using Generalized Advantage Estimation.

    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)

    Return: shape (t, env)
    '''
    advantages = t.zeros_like(rewards).to(device)

    last_adv = 0
    for i in reversed(range(rewards.shape[0])):
        done_to_use = dones[i + 1] if i + 1 < rewards.shape[0] else next_done
        value_to_use = values[i + 1] if i + 1 < rewards.shape[0] else next_value

        delta = -values[i] + rewards[i] + gamma * (1 - done_to_use) * value_to_use
        advantages[i] = last_adv = delta + gamma * gae_lambda * (1 - done_to_use) * last_adv
    
    return advantages

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_compute_advantages(compute_advantages)

# %%
@dataclass
class Minibatch:
    obs: t.Tensor
    logprobs: t.Tensor
    actions: t.Tensor
    advantages: t.Tensor
    returns: t.Tensor
    values: t.Tensor

def minibatch_indexes(batch_size: int, minibatch_size: int) -> List[np.ndarray]:
    '''Return a list of length (batch_size // minibatch_size) where each element is an array of indexes into the batch.

    Each index should appear exactly once.
    '''
    assert batch_size % minibatch_size == 0
    print(batch_size, minibatch_size, batch_size // minibatch_size)

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_minibatch_indexes(minibatch_indexes)

def make_minibatches(
    obs: t.Tensor,
    logprobs: t.Tensor,
    actions: t.Tensor,
    advantages: t.Tensor,
    values: t.Tensor,
    obs_shape: tuple,
    action_shape: tuple,
    batch_size: int,
    minibatch_size: int,
) -> list[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    pass
# %%
