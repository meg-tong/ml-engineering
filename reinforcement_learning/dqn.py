#%%
import argparse
import os
import random
import re
import sys
import time
from dataclasses import dataclass
from distutils.util import strtobool
from typing import Any, Iterable, List, Optional, Tuple, Union

import gym
import gym.envs.registration
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import rearrange
from gym.spaces import Box, Discrete
from matplotlib import pyplot as plt

import wandb
from numpy.random import Generator
from torch.utils.tensorboard import SummaryWriter

import rl_utils

MAIN = __name__ == "__main__"
os.environ["SDL_VIDEODRIVER"] = "dummy"

#%%
class QNetwork(nn.Module):
    def __init__(self, dim_observation: int, num_actions: int, hidden_sizes: List[int] = [120, 84]):
        super().__init__()
        self.linear1 = nn.Linear(dim_observation, hidden_sizes[0])
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_sizes[1], num_actions)

    def forward(self, x: t.Tensor) -> t.Tensor:
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x

if MAIN:
    net = QNetwork(dim_observation=4, num_actions=2)
    n_params = sum((p.nelement() for p in net.parameters()))
    print(net)
    print(f"Total number of parameters: {n_params}")
    assert n_params == 10934
# %%
@dataclass
class ReplayBufferSamples:
    '''
    Samples from the replay buffer, converted to PyTorch for use in neural network training.
    obs: shape (sample_size, *observation_shape), dtype t.float
    actions: shape (sample_size, ) dtype t.int
    rewards: shape (sample_size, ), dtype t.float
    dones: shape (sample_size, ), dtype t.bool
    next_observations: shape (sample_size, *observation_shape), dtype t.float
    '''
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

class ReplayBuffer:
    rng: Generator
    observations: t.Tensor
    actions: t.Tensor
    rewards: t.Tensor
    dones: t.Tensor
    next_observations: t.Tensor

    def __init__(self, buffer_size: int, num_actions: int, observation_shape: tuple, num_environments: int, seed: int):
        assert num_environments == 1, "This buffer only supports SyncVectorEnv with 1 environment inside."
        self.buffer_size = buffer_size
        self.observations = t.zeros(size=(buffer_size, *observation_shape), dtype=t.float64)
        self.actions = t.zeros((buffer_size), dtype=t.int64)
        self.rewards = t.zeros((buffer_size), dtype=t.float64)
        self.dones = t.zeros((buffer_size), dtype=t.bool)
        self.next_observations = t.zeros((buffer_size, *observation_shape), dtype=t.float64)

        self.rng = np.random.default_rng(seed)
        self.total = 0

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
        '''
        obs: shape (num_environments, *observation_shape) 
            Observation before the action
        actions: shape (num_environments, ) 
            Action chosen by the agent
        rewards: shape (num_environments, ) 
            Reward after the action
        dones: shape (num_environments, ) 
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) 
            Observation after the action
            If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        '''
        index = self.total % self.buffer_size
        self.observations[index, ...] = t.tensor(obs)
        self.actions[index, ...] = t.tensor(actions)
        self.rewards[index, ...] = t.tensor(rewards)
        self.dones[index, ...] = t.tensor(dones)
        self.next_observations[index, ...] = t.tensor(next_obs)

        self.total += 1

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        sample_indices = self.rng.integers(low=0, high=min(self.total, self.buffer_size), size=sample_size)
        obs = self.observations[sample_indices, ...].to(device)
        act =  self.actions[sample_indices, ...].to(device)
        rew = self.rewards[sample_indices, ...].to(device)
        don = self.dones[sample_indices, ...].to(device)
        nex_obs = self.next_observations[sample_indices, ...].to(device)

        return ReplayBufferSamples(obs, act, rew, don, nex_obs)

if MAIN:
    rl_utils.test_replay_buffer_single(ReplayBuffer)
    rl_utils.test_replay_buffer_deterministic(ReplayBuffer)
    rl_utils.test_replay_buffer_wraparound(ReplayBuffer)

# %%
if MAIN:
    rb = ReplayBuffer(buffer_size=256, num_actions=2, observation_shape=(4,), num_environments=1, seed=0)
    envs = gym.vector.SyncVectorEnv([rl_utils.make_env("CartPole-v1", 0, 0, False, "test")])
    obs = envs.reset()
    for i in range(512):
        actions = np.array([0])
        (next_obs, rewards, dones, infos) = envs.step(actions)
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
    sample = rb.sample(128, t.device("cpu"))
    columns = ["cart_pos", "cart_v", "pole_angle", "pole_v"]
    df = pd.DataFrame(rb.observations, columns=columns)
    df.plot(subplots=True, title="Replay Buffer")
    df2 = pd.DataFrame(sample.observations, columns=columns)
    df2.plot(subplots=True, title="Shuffled Replay Buffer")
# %%
def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    return end_e + (start_e - end_e) * max(0, 1 - current_step / (exploration_fraction * total_timesteps))

if MAIN:
    epsilons = [
        linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)
    ]
    rl_utils.test_linear_schedule(linear_schedule)
# %%
def epsilon_greedy_policy(
    envs: gym.vector.SyncVectorEnv, q_network: QNetwork, rng: Generator, obs: t.Tensor, epsilon: float
) -> np.ndarray:
    '''With probability epsilon, take a random action. Otherwise, take a greedy action according to the q_network.
    Inputs:
        envs : gym.vector.SyncVectorEnv, the family of environments to run against
        q_network : QNetwork, the network used to approximate the Q-value function
        obs : The current observation
        epsilon : exploration percentage
    Outputs:
        actions: (n_environments, ) the sampled action for each environment.
    '''
    if rng.random() <= epsilon:
        return rng.integers(low=0, high=envs.single_action_space.n, size=envs.num_envs)
    return q_network(obs).argmax(dim=-1).detach().numpy()

if MAIN:
    rl_utils.test_epsilon_greedy_policy(epsilon_greedy_policy)
# %%
ObsType = np.ndarray
ActType = int

class Probe1(gym.Env):
    '''One action, observation of [0.0], one timestep long, +1 reward.

    We expect the agent to rapidly learn that the value of the constant [0.0] observation is +1.0. Note we're using a continuous observation space for consistency with CartPole.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([0]), np.array([0]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return (np.array([0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])

gym.envs.registration.register(id="Probe1-v0", entry_point=Probe1)
if MAIN:
    env = gym.make("Probe1-v0")
    assert env.observation_space.shape == (1,)
    assert env.action_space.shape == ()
# %%
class Probe2(gym.Env):
    '''One action, observation of [-1.0] or [+1.0], one timestep long, reward equals observation.

    We expect the agent to rapidly learn the value of each observation is equal to the observation.
    '''

    action_space: Discrete
    observation_space: Box
    observation: float

    def __init__(self):
        super().__init__()
        self.observation_space = Box(np.array([-1]), np.array([1]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return (np.array([self.observation]), self.observation, True, {})
        
    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        self.observation = -1.0 if np.random.rand() < 0.5 else 1.0
        if return_info:
            return (np.array([self.observation]), {})
        return np.array([self.observation])

gym.envs.registration.register(id="Probe2-v0", entry_point=Probe2)

class Probe3(gym.Env):
    '''One action, [0.0] then [1.0] observation, two timesteps, +1 reward at the end.

    We expect the agent to rapidly learn the discounted value of the initial observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        self.observation_space = Box(np.array([0]), np.array([1]))
        self.action_space = Discrete(1)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return (np.array([1.0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])

gym.envs.registration.register(id="Probe3-v0", entry_point=Probe3)

class Probe4(gym.Env):
    '''Two actions, [0.0] observation, one timestep, reward is -1.0 or +1.0 dependent on the action.

    We expect the agent to learn to choose the +1.0 action.
    '''
    action_space: Discrete
    observation_space: Box

    def __init__(self):
        self.observation_space = Box(np.array([0]), np.array([1]))
        self.action_space = Discrete(2)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        if action == 0:
            return (np.array([0.0]), -1.0, True, {})
        return (np.array([0.0]), 1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        
        if return_info:
            return (np.array([0.0]), {})
        return np.array([0.0])

gym.envs.registration.register(id="Probe4-v0", entry_point=Probe4)

class Probe5(gym.Env):
    '''Two actions, random 0/1 observation, one timestep, reward is 1 if action equals observation otherwise -1.

    We expect the agent to learn to match its action to the observation.
    '''

    action_space: Discrete
    observation_space: Box

    def __init__(self):
        self.observation_space = Box(np.array([0]), np.array([1]))
        self.action_space = Discrete(2)
        self.seed()
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        return (np.array([self.observation]), 1.0 if self.observation == action else -1.0, True, {})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        
        self.observation = 0.0 if np.random.rand() < 0.5 else 1.0
        if return_info:
            return (np.array([self.observation]), {})
        return np.array([self.observation])

gym.envs.registration.register(id="Probe5-v0", entry_point=Probe5)
# %%
@dataclass
class DQNArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "DQN_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = ""
    wandb_entity: Optional[str] = None
    capture_video: bool = True
    env_id: str = "CartPole-v1" # Probe1-v0
    total_timesteps: int = 500000
    learning_rate: float = 0.00025
    buffer_size: int = 10000
    gamma: float = 0.99
    target_network_frequency: int = 500
    batch_size: int = 128
    start_e: float = 1.0
    end_e: float = 0.05
    exploration_fraction: float = 0.5
    learning_starts: int = 10000
    train_frequency: int = 10

arg_help_strings = dict(
    exp_name = "the name of this experiment",
    seed = "seed of the experiment",
    torch_deterministic = "if toggled, `torch.backends.cudnn.deterministic=False`",
    cuda = "if toggled, cuda will be enabled by default",
    track = "if toggled, this experiment will be tracked with Weights and Biases",
    wandb_project_name = "the wandb's project name",
    wandb_entity = "the entity (team) of wandb's project",
    capture_video = "whether to capture videos of the agent performances (check out `videos` folder)",
    env_id = "the id of the environment",
    total_timesteps = "total timesteps of the experiments",
    learning_rate = "the learning rate of the optimizer",
    buffer_size = "the replay memory buffer size",
    gamma = "the discount factor gamma",
    target_network_frequency = "the timesteps it takes to update the target network",
    batch_size = "the batch size of samples from the replay memory",
    start_e = "the starting epsilon for exploration",
    end_e = "the ending epsilon for exploration",
    exploration_fraction = "the fraction of `total-timesteps` it takes from start-e to go end-e",
    learning_starts = "timestep to start learning",
    train_frequency = "the frequency of training",
)
toggles = ["torch_deterministic", "cuda", "track", "capture_video"]

def parse_args(arg_help_strings=arg_help_strings, toggles=toggles) -> DQNArgs:
    parser = argparse.ArgumentParser()
    for (name, field) in DQNArgs.__dataclass_fields__.items():
        flag = "--" + name.replace("_", "-")
        type_function = field.type if field.type != bool else lambda x: bool(strtobool(x))
        toggle_kwargs = {"nargs": "?", "const": True} if name in toggles else {}
        parser.add_argument(
            flag, type=type_function, default=field.default, help=arg_help_strings[name], **toggle_kwargs
        )
    return DQNArgs(**vars(parser.parse_args()))

def setup(args: DQNArgs) -> Tuple[str, SummaryWriter, np.random.Generator, t.device, gym.vector.SyncVectorEnv]:
    '''Helper function to set up useful variables for the DQN implementation'''
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name if args.wandb_project_name != "" else "meg-" + args.env_id,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % "\n".join([f"|{key}|{value}|" for (key, value) in vars(args).items()]),
    )
    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    t.backends.cudnn.deterministic = args.torch_deterministic
    rng = np.random.default_rng(args.seed)
    device = t.device("cuda" if t.cuda.is_available() and args.cuda else "cpu")
    envs = gym.vector.SyncVectorEnv([rl_utils.make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, Discrete), "only discrete action space is supported"
    return (run_name, writer, rng, device, envs)

def log(
    track: bool,
    writer: SummaryWriter,
    start_time: float,
    step: int,
    predicted_q_vals: t.Tensor,
    loss: Union[float, t.Tensor],
    infos: Iterable[dict],
    epsilon: float,
):
    '''Helper function to write relevant info to TensorBoard logs, and print some things to stdout'''
    if step % 100 == 0:
        writer.add_scalar("losses/td_loss", loss, step)
        writer.add_scalar("losses/q_values", predicted_q_vals.mean().item(), step)
        writer.add_scalar("charts/SPS", int(step / (time.time() - start_time)), step)
        if step % 10000 == 0:
            print("SPS:", int(step / (time.time() - start_time)))
    for info in infos:
        if "episode" in info.keys():
            print(f"global_step={step}, episodic_return={info['episode']['r']}")
            writer.add_scalar("charts/episodic_return", info["episode"]["r"], step)
            writer.add_scalar("charts/episodic_length", info["episode"]["l"], step)
            writer.add_scalar("charts/epsilon", epsilon, step)
            if track:
                wandb.log({"episodic_return": info['episode']['r'], "episodic_length": info['episode']['l'], "epsilon": epsilon}, step=step)
            break
    if track:
        wandb.log({"td_loss": loss, "q_values": predicted_q_vals.mean().item()}, step=step)
# %%
def train_dqn(args: DQNArgs):
    (run_name, writer, rng, device, envs) = setup(args)
    num_actions = envs.single_action_space.n
    observation_shape = envs.observation_space.shape # observation shape ??

    q_network = QNetwork(dim_observation=observation_shape[1], num_actions=num_actions).to(device)
    q_network_target = QNetwork(dim_observation=observation_shape[1], num_actions=num_actions).to(device)
    q_network_target.load_state_dict(q_network.state_dict()) # Is this a shallow or deep copy ?
    optimizer = t.optim.Adam(q_network.parameters(), lr=args.learning_rate)
    rb = ReplayBuffer(
        buffer_size=args.buffer_size, 
        num_actions=num_actions, 
        observation_shape=observation_shape,  
        num_environments=envs.num_envs,
        seed=args.seed)

    start_time = time.time()
    obs = envs.reset()
    for step in range(args.total_timesteps):
        epsilon = linear_schedule(step, args.start_e, args.end_e, args.exploration_fraction, args.total_timesteps)
        actions = epsilon_greedy_policy(envs=envs, q_network=q_network, rng=rng, obs=t.tensor(obs), epsilon=epsilon)
        (next_obs, rewards, dones, infos) = envs.step(actions)

        "Boilerplate to handle the terminal observation case"
        real_next_obs = next_obs.copy()
        for (i, done) in enumerate(dones):
            if done:
                real_next_obs[i] = infos[i]["terminal_observation"]
        rb.add(obs, actions, rewards, dones, next_obs)
        obs = next_obs
        if step > args.learning_starts and step % args.train_frequency == 0:
            rbs = rb.sample(args.batch_size, device=device)

            with t.inference_mode(): # 0 where dones is True
                not_dones = 1 - rbs.dones.float()
                target_next_q = q_network_target(rbs.next_observations.squeeze().float())
                y = rbs.rewards + args.gamma * not_dones * target_next_q.max(-1).values # take max over last dimension ?
            predicted_q_vals = q_network(rbs.observations.squeeze().float())[t.arange(args.batch_size), rbs.actions] # t.gather 2D
            loss = t.pow(y - predicted_q_vals, 2).sum() / args.batch_size
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            log(args.track, writer, start_time, step, predicted_q_vals, loss, infos, epsilon)

            if step % args.train_frequency == 0:
                q_network_target.load_state_dict(q_network.state_dict())

    "If running one of the Probe environments, will test if the learned q-values are sensible after training. Useful for debugging."
    probe_batches = [
        t.tensor([[0.0]]), t.tensor([[-1.0], [+1.0]]), t.tensor([[0.0], [1.0]]), t.tensor([[0.0]]), t.tensor([[0.0], [1.0]])
    ]
    if re.match(r"Probe(\d)-v0", args.env_id):
        probe_no = int(re.match(r"Probe(\d)-v0", args.env_id).group(1))
        batch = probe_batches[probe_no]
        value = q_network(batch)
        print("Value: ", value)
        expected = t.tensor([[1.0]]).to(device)
        t.testing.assert_close(value, expected, 0.0001, 0.0)

    envs.close()
    writer.close()

if MAIN:
    if "ipykernel_launcher" in os.path.basename(sys.argv[0]):
        filename = globals().get("__file__", "<filename of this script>")
        print(f"Try running this file from the command line instead: python {os.path.basename(filename)} --help")
        args = DQNArgs()
    else:
        args = parse_args()
    train_dqn(args)
# %%
