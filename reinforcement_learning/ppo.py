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

        delta = -values[i] + rewards[i] + gamma * (1 - done_to_use.float()) * value_to_use
        advantages[i] = last_adv = delta + gamma * gae_lambda * (1 - done_to_use.float()) * last_adv
    
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
    batch_indices = np.random.permutation(batch_size)

    return [batch_indices[x:x+minibatch_size] for x in range(0, len(batch_indices), minibatch_size)]

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
) -> List[Minibatch]:
    '''Flatten the environment and steps dimension into one batch dimension, then shuffle and split into minibatches.'''
    obs = rearrange(obs, 't env obs_space-> (t env) obs_space')
    logprobs = rearrange(logprobs, 't env -> (t env)')
    actions = rearrange(actions, 't env -> (t env)')
    advantages = rearrange(advantages, 't env -> (t env)')
    values = rearrange(values, 't env -> (t env)')

    mb_indexes = minibatch_indexes(batch_size, minibatch_size)
    return [Minibatch(obs[mb], logprobs[mb], actions[mb], advantages[mb], advantages[mb] + values[mb], values[mb]) for mb in mb_indexes]

#%%
def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float, normalize: bool = True
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    if normalize:
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-12)
    actor_logprobs = probs.log_prob(mb_action)
    ratio = t.exp(actor_logprobs - mb_logprobs)
    clipped_mb_advantages = t.where(mb_advantages >= 0, (1 + clip_coef) * mb_advantages, (1 - clip_coef) * mb_advantages)
    return t.sum(t.min(ratio * mb_advantages, clipped_mb_advantages)) / len(mb_action)

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_calc_policy_loss(calc_policy_loss, verbose=False)
    
# %%
def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, v_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    v_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    return v_coef * 0.5 * nn.MSELoss()(critic(mb_obs), mb_returns)

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_calc_value_function_loss(calc_value_function_loss, verbose=False)

#%%
def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()

if MAIN and RUNNING_FROM_FILE:
    rl_utils.test_calc_entropy_loss(calc_entropy_loss, verbose=False)
#%%
class PPOScheduler:
    def __init__(self, optimizer, initial_lr: float, end_lr: float, num_updates: int):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.end_lr = end_lr
        self.num_updates = num_updates
        self.n_step_calls = 0

    def step(self):
        '''Implement linear learning rate decay so that after num_updates calls to step, the learning rate is end_lr.'''
        return self.end_lr + min(0, self.num_updates - self.n_step_calls) * (self.initial_lr - self.end_lr)

def make_optimizer(agent: Agent, num_updates: int, initial_lr: float, end_lr: float) -> Tuple[optim.Adam, PPOScheduler]:
    '''Return an appropriately configured Adam with its attached scheduler.'''
    adam = optim.Adam(agent.parameters(), lr=initial_lr, eps=1e-5)
    ppo_scheduler = PPOScheduler(adam, initial_lr=initial_lr, end_lr=end_lr, num_updates=num_updates)
    return (adam, ppo_scheduler)

#%%
@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(__file__).rstrip(".py")
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False#
    wandb_project_name: str = "meg-PPOCart"#
    wandb_entity: str = None
    capture_video: bool = False
    env_id: str = "CartPole-v1"
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    num_envs: int = 4
    num_steps: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    clip_coef: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    batch_size: int = 512
    minibatch_size: int = 128

def train_ppo(args):
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb
        #wandb.init(settings=wandb.Settings(start_method="fork"))
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    t.backends.cudnn.deterministic = args.torch_deterministic
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv(
        [rl_utils.make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    action_shape = envs.single_action_space.shape
    assert action_shape is not None
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    agent = Agent(envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    (optimizer, scheduler) = make_optimizer(agent, num_updates, args.learning_rate, 0.0)
    obs = t.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = t.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = t.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = t.zeros((args.num_steps, args.num_envs)).to(device)
    dones = t.zeros((args.num_steps, args.num_envs)).to(device)
    values = t.zeros((args.num_steps, args.num_envs)).to(device)
    global_step = 0
    old_approx_kl = 0.0
    approx_kl = 0.0
    value_loss = t.tensor(0.0)
    policy_loss = t.tensor(0.0)
    entropy_loss = t.tensor(0.0)
    clipfracs = []
    info = []
    start_time = time.time()
    next_obs = t.Tensor(envs.reset()).to(device)
    next_done = t.zeros(args.num_envs).to(device)
    for global_step in range(num_updates):
        for i in range(0, args.num_steps): # need to do this in inference mode?
            # Check next_obs/next_done/next_values are all calculated in the right order
            obs[i] = next_obs
            dones[i] = next_done
            with t.inference_mode():
                next_values = agent.critic(obs[i])
                logits = agent.actor(obs[i])
            probs = t.distributions.categorical.Categorical(logits=logits)
            actions[i] = probs.sample()
            logprobs[i] = probs.log_prob(actions[i])

            (next_obs, reward, next_done, info) = envs.step(actions[i].numpy().astype('long'))
            next_obs = t.tensor(next_obs)
            next_done = t.tensor(next_done)
            rewards[i] = t.tensor(reward).to(device)
            values[i] = next_values.squeeze()
            
            for item in info:
                if "episode" in item.keys() and global_step % 10 == 0:
                    print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                    if args.track:
                        wandb.log({"episodic_return": item["episode"]["r"]}, step=global_step)
                        wandb.log({"episodic_repisodic_lengtheturn": item["episode"]["l"]}, step=global_step)
                    break
        next_value = rearrange(agent.critic(next_obs), "env 1 -> 1 env")
        advantages = compute_advantages(
            next_value, next_done, rewards, values, dones, device, args.gamma, args.gae_lambda
        )
        clipfracs.clear()
        for _ in range(args.update_epochs):
            minibatches = make_minibatches(
                obs,
                logprobs,
                actions,
                advantages,
                values,
                envs.single_observation_space.shape,
                action_shape,
                args.batch_size,
                args.minibatch_size,
            )
            for mb in minibatches:
                optimizer.zero_grad()
                
                logits = agent.actor(mb.obs)
                probs = t.distributions.categorical.Categorical(logits=logits)
                policy_loss = calc_policy_loss(probs, mb.actions, mb.advantages, mb.logprobs, args.clip_coef)
                value_loss = calc_value_function_loss(agent.critic, mb.obs, mb.returns, args.vf_coef)
                entropy_loss = calc_entropy_loss(probs, args.ent_coef)
                loss = policy_loss + value_loss + entropy_loss # should we be summing over the loss across multiple minibatches
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) 
                optimizer.step()
        
        scheduler.step()
        (y_pred, y_true) = (mb.values.cpu().numpy(), mb.returns.cpu().numpy())
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        with t.no_grad():
            newlogprob: t.Tensor = probs.log_prob(mb.actions)
            logratio = newlogprob - mb.logprobs
            ratio = logratio.exp()
            old_approx_kl = (-logratio).mean().item()
            approx_kl = (ratio - 1 - logratio).mean().item()
            clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
        if args.track:
            wandb.log({"learning_rate": optimizer.param_groups[0]["lr"]}, step=global_step)
            wandb.log({"value_loss": value_loss.item()}, step=global_step)
            wandb.log({"policy_loss": policy_loss.item()}, step=global_step)
            wandb.log({"entropy": entropy_loss.item()}, step=global_step)
            wandb.log({"old_approx_kl": old_approx_kl}, step=global_step)
            wandb.log({"approx_kl": approx_kl}, step=global_step)
            wandb.log({"clipfrac": np.mean(clipfracs)}, step=global_step)
            wandb.log({"explained_variance": explained_var}, step=global_step)
            wandb.log({"SPS": int(global_step / (time.time() - start_time))}, step=global_step)
        if global_step % 10 == 0:
            print("steps per second (SPS):", int(global_step / (time.time() - start_time)))
    envs.close()

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = PPOArgs()
    else:
        args = rl_utils.ppo_parse_args()
    train_ppo(args)
# %%
