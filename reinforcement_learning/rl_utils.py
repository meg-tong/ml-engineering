import argparse
import os
import random
from dataclasses import asdict, dataclass
from distutils.util import strtobool
from typing import Any, List, Optional, Tuple, Union

import gym
import gym.envs.registration
import gym.spaces
import numpy as np
import torch
import torch as t
import torch.nn as nn
from fancy_einsum import einsum
from numpy.random import Generator
from PIL import Image, ImageDraw
from torch import nn
from torch.distributions.categorical import Categorical

Arr = np.ndarray

ObsType = int
ActType = int

class Environment:
    def __init__(self, num_states: int, num_actions: int, start=0, terminal=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.start = start
        self.terminal = np.array([], dtype=int) if terminal is None else terminal
        (self.T, self.R) = self.build()

    def build(self):
        '''
        Constructs the T and R tensors from the dynamics of the environment.
        Outputs:
            T : (num_states, num_actions, num_states) State transition probabilities
            R : (num_states, num_actions, num_states) Reward function
        '''
        num_states = self.num_states
        num_actions = self.num_actions
        T = np.zeros((num_states, num_actions, num_states))
        R = np.zeros((num_states, num_actions, num_states))
        for s in range(num_states):
            for a in range(num_actions):
                (states, rewards, probs) = self.dynamics(s, a)
                (all_s, all_r, all_p) = self.out_pad(states, rewards, probs)
                T[s, a, all_s] = all_p
                R[s, a, all_s] = all_r
        return (T, R)

    def dynamics(self, state: int, action: int) -> Tuple[Arr, Arr, Arr]:
        '''
        Computes the distribution over possible outcomes for a given state
        and action.
        Inputs:
            state : int (index of state)
            action : int (index of action)
        Outputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        '''
        raise NotImplementedError

    def render(pi: Arr):
        '''
        Takes a policy pi, and draws an image of the behavior of that policy,
        if applicable.
        Inputs:
            pi : (num_actions,) a policy
        Outputs:
            None
        '''
        raise NotImplementedError

    def out_pad(self, states: Arr, rewards: Arr, probs: Arr):
        '''
        Inputs:
            states  : (m,) all the possible next states
            rewards : (m,) rewards for each next state transition
            probs   : (m,) likelihood of each state-reward pair
        Outputs:
            states  : (num_states,) all the next states
            rewards : (num_states,) rewards for each next state transition
            probs   : (num_states,) likelihood of each state-reward pair (including
                           probability zero outcomes.)
        '''
        out_s = np.arange(self.num_states)
        out_r = np.zeros(self.num_states)
        out_p = np.zeros(self.num_states)
        for i in range(len(states)):
            idx = states[i]
            out_r[idx] += rewards[i]
            out_p[idx] += probs[i]
        return (out_s, out_r, out_p)

class Norvig(Environment):
    def dynamics(self, state : int, action : int) -> Tuple[Arr, Arr, Arr]:
        def state_index(state):
            assert 0 <= state[0] < self.width and 0 <= state[1] < self.height, print(state)
            pos = state[0] + state[1] * self.width
            assert 0 <= pos < self.num_states, print(state, pos)
            return pos

        pos = self.states[state]
        move = self.actions[action]

        # When in either goal state (or the wall), stay there forever, no reward
        if state in self.terminal or state in self.walls:
            return (np.array([state]), np.array([0]), np.array([1]))

        # 70% chance of moving in correct direction
        # 10% chance of moving in the other directions
        out_probs = np.zeros(self.num_actions) + 0.1  # set slippery probability
        out_probs[action] = 0.7  # probability of requested direction

        out_states = np.zeros(self.num_actions, dtype=int) + self.num_actions
        out_rewards = np.zeros(self.num_actions) + self.penalty
        new_states = [pos + x for x in self.actions]

        for i, s_new in enumerate(new_states):

            # check if left bounds of world, if so, don't move
            if not (0 <= s_new[0] < self.width and 0 <= s_new[1] < self.height):
                out_states[i] = state
                continue

            # position in bounds, lookup state index
            new_state = state_index(s_new)  # lookup state index

            # check if would run into a wall, if so, don't move
            if new_state in self.walls:
                out_states[i] = state

            # a normal movement, move to new cell
            else:
                out_states[i] = new_state

            # walking into a goal state from non-goal state
            for idx in range(len(self.terminal)):
                if new_state == self.terminal[idx]:
                    out_rewards[i] = self.goal_rewards[idx]

        return (out_states, out_rewards, out_probs)

    def render(self, pi: Arr):
        pi = pi.reshape((3, 4))
        objects = {(3, 0): "green", (1, 1): "black", (3, 1): "red"}
        img = Image.new(mode="RGB", size=(400, 300), color="white")
        draw = ImageDraw.Draw(img)
        for x in range(0, img.width+1, 100):
            draw.line([(x, 0), (x, img.height)], fill="black", width=4)
        for y in range(0, img.height+1, 100):
            draw.line([(0, y), (img.width, y)], fill="black", width=4)
        for x in range(4):
            for y in range(3):
                bounds = (50+x*100, 50+y*100)
                draw.regular_polygon((*bounds, 20), 3, rotation=-int(90*pi[y][x]), fill="black")
                if (x, y) in objects:
                    draw.regular_polygon((*bounds, 40), 4, fill=objects[(x, y)])
        img.show()

    def __init__(self, penalty=-0.04):

        self.height = 3
        self.width = 4
        self.penalty = penalty
        num_states = self.height * self.width
        num_actions = 4
        self.states = np.array([[x, y] for y in range(self.height) for x in range(self.width)])
        self.actions = np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])  # up, right, down, left
        self.dim = (self.height, self.width)

        # special states: Tuples of state and reward
        # all other states get penalty
        start = 8
        terminal = np.array([3, 7], dtype=int)
        self.walls = np.array([5], dtype=int)
        self.goal_rewards = np.array([1.0, -1.0])

        super().__init__(num_states, num_actions, start=start, terminal=terminal)

def find_optimal_policy(env: Environment, gamma=0.99, max_iterations=10_000):
    '''
    Inputs:
        env: environment
    Outputs:
        pi : (num_states,) int, of actions represeting an optimal policy
    '''
    pi = np.zeros(shape=env.num_states, dtype=int)

    for i in range(max_iterations):
        V = policy_eval_exact(env, pi, gamma)
        pi_new = policy_improvement(env, V, gamma)
        if np.array_equal(pi_new, pi):
            return pi_new
        else:
            pi = pi_new
    else:
        print(f"Failed to converge after {max_iterations} steps.")
        return pi

def policy_eval_numerical(env: Environment, pi: Arr, gamma=0.99, eps=1e-08, max_iterations=10_000) -> Arr:
    '''
    Numerically evaluates the value of a given policy by iterating the Bellman equation
    Inputs:
        env: Environment
        pi : shape (num_states,) - The policy to evaluate
        gamma: float - Discount factor
        eps  : float - Tolerance
    Outputs:
        value : float (num_states,) - The value function for policy pi
    '''
    # Indexing T into an array of shape (num_states, num_states)
    states = np.arange(env.num_states)
    actions = pi
    transition_matrix = env.T[states, actions, :]
    # Same thing with R
    reward_matrix = env.R[states, actions, :]
    
    # Iterate until we get convergence
    V = np.zeros_like(pi)
    for i in range(max_iterations):
        V_new = einsum("s s_prime, s s_prime -> s", transition_matrix, reward_matrix + gamma * V)
        if np.abs(V - V_new).max() < eps:
            print(f"Converged in {i} steps.")
            return V_new
        V = V_new
    print(f"Failed to converge in {max_iterations} steps.")
    return V

def policy_eval_exact(env: Environment, pi: Arr, gamma=0.99) -> Arr:
    '''
    Finds the exact solution to the Bellman equation.
    '''
    states = np.arange(env.num_states)
    actions = pi
    transition_matrix = env.T[states, actions, :]
    reward_matrix = env.R[states, actions, :]

    r = einsum("i j, i j -> i", transition_matrix, reward_matrix)

    mat = np.eye(env.num_states) - gamma * transition_matrix

    return np.linalg.solve(mat, r)


def policy_improvement(env: Environment, V: Arr, gamma=0.99) -> Arr:
    '''
    Inputs:
        env: Environment
        V  : (num_states,) value of each state following some policy pi
    Outputs:
        pi_better : vector (num_states,) of actions representing a new policy obtained via policy iteration
    '''
    states = np.arange(env.num_states)
    transition_matrix = env.T[states, :, :]
    reward_matrix = env.R[states, :, :]
    
    V_for_each_action = einsum("s a s_prime, s a s_prime -> s a", transition_matrix, reward_matrix + gamma * V)
    pi_better = V_for_each_action.argmax(-1)

    return pi_better

# Alternate solution:

def policy_improvement_2(env : Environment, V : Arr, gamma=0.99) -> Arr:
    pi_new = np.argmax(np.einsum("ijk,ijk -> ij", env.T, env.R) + gamma * np.einsum("ijk,k -> ij", env.T, V), axis=1)
    return pi_new

class DiscreteEnviroGym(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete

    def __init__(self, env: Environment):
        super().__init__()
        self.env = env
        self.observation_space = gym.spaces.Discrete(env.num_states)
        self.action_space = gym.spaces.Discrete(env.num_actions)
        self.reset()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        '''
        Samples from the underlying dynamics of the environment
        '''
        (states, rewards, probs) = self.env.dynamics(self.pos, action)
        idx = self.np_random.choice(len(states), p=probs)
        (new_state, reward) = (states[idx], rewards[idx])
        self.pos = new_state
        done = self.pos in self.env.terminal
        return (new_state, reward, done, {"env": self.env})

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        self.pos = self.env.start
        return (self.pos, {"env": self.env}) if return_info else self.pos

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"

class QNetwork(nn.Module):
    def __init__(self, dim_observation: int, num_actions: int, hidden_sizes: List[int] = [120, 84]):
        super().__init__()
        in_features_list = [dim_observation] + hidden_sizes
        out_features_list = hidden_sizes + [num_actions]
        layers = []
        for i, (in_features, out_features) in enumerate(zip(in_features_list, out_features_list)):
            layers.append(nn.Linear(in_features, out_features))
            if i < len(in_features_list) - 1:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.layers(x)

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

# %%
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
        self.rng = np.random.default_rng(seed)
        self.buffer = [None for i in range(5)]

    def add(
        self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray, next_obs: np.ndarray
    ) -> None:
        '''
        obs: shape (num_environments, *observation_shape) 
            Observation before the action
        actions: shape (num_environments,) 
            Action chosen by the agent
        rewards: shape (num_environments,) 
            Reward after the action
        dones: shape (num_environments,) 
            If True, the episode ended and was reset automatically
        next_obs: shape (num_environments, *observation_shape) 
            Observation after the action
            If done is True, this should be the terminal observation, NOT the first observation of the next episode.
        '''
        for i, (arr, arr_list) in enumerate(zip([obs, actions, rewards, dones, next_obs], self.buffer)):
            if arr_list is None:
                self.buffer[i] = arr
            else:
                self.buffer[i] = np.concatenate((arr, arr_list))
            if self.buffer[i].shape[0] > self.buffer_size:
                self.buffer[i] = self.buffer[i][:self.buffer_size]

        self.observations, self.actions, self.rewards, self.dones, self.next_observations = [t.as_tensor(arr) for arr in self.buffer]

    def sample(self, sample_size: int, device: t.device) -> ReplayBufferSamples:
        '''
        Uniformly sample sample_size entries from the buffer and convert them to PyTorch tensors on device.
        Sampling is with replacement, and sample_size may be larger than the buffer size.
        '''
        indices = self.rng.integers(0, self.buffer[0].shape[0], sample_size)
        samples = [t.as_tensor(arr_list[indices], device=device) for arr_list in self.buffer]
        return ReplayBufferSamples(*samples)

def linear_schedule(
    current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int
) -> float:
    '''Return the appropriate epsilon for the current step.

    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).

    It should stay at end_e for the rest of the episode.
    '''
    return start_e + (end_e - start_e) * min(current_step / (exploration_fraction * total_timesteps), 1)

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
    num_actions = envs.single_action_space.n
    if rng.random() < epsilon:
        return rng.integers(0, num_actions, size = (envs.num_envs,))
    else:
        q_scores = q_network(obs)
        return q_scores.argmax(-1).detach().cpu().numpy()

@dataclass
class PPOArgs:
    exp_name: str = os.path.basename(globals().get("__file__", "PPO_implementation").rstrip(".py"))
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True
    wandb_project_name: str = "PPOCart"
    wandb_entity: str = None
    capture_video: bool = True
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
    """Compute advantages using Generalized Advantage Estimation.
    next_value: shape (1, env) - represents V(s_{t+1}) which is needed for the last advantage term
    next_done: shape (env,)
    rewards: shape (t, env)
    values: shape (t, env)
    dones: shape (t, env)
    Return: shape (t, env)
    """
    "SOLUTION"
    T = values.shape[0]
    next_values = torch.concat([values[1:], next_value])
    next_dones = torch.concat([dones[1:], next_done.unsqueeze(0)])
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    advantages = deltas.clone().to(device)
    for t in reversed(range(1, T)):
        advantages[t-1] = deltas[t-1] + gamma * gae_lambda * (1.0 - dones[t]) * advantages[t]
    return advantages

def calc_policy_loss(
    probs: Categorical, mb_action: t.Tensor, mb_advantages: t.Tensor, mb_logprobs: t.Tensor, clip_coef: float
) -> t.Tensor:
    '''Return the policy loss, suitable for maximisation with gradient ascent.

    probs: a distribution containing the actor's unnormalized logits of shape (minibatch, num_actions)

    clip_coef: amount of clipping, denoted by epsilon in Eq 7.

    normalize: if true, normalize mb_advantages to have mean 0, variance 1
    '''
    logits_diff = (probs.log_prob(mb_action) - mb_logprobs)

    r_theta = t.exp(logits_diff)

    mb_advantages = (mb_advantages - mb_advantages.mean()) / mb_advantages.std()

    non_clipped = r_theta * mb_advantages
    clipped = t.clip(r_theta, 1-clip_coef, 1+clip_coef) * mb_advantages

    return t.minimum(non_clipped, clipped).mean()


def calc_value_function_loss(critic: nn.Sequential, mb_obs: t.Tensor, mb_returns: t.Tensor, vf_coef: float) -> t.Tensor:
    '''Compute the value function portion of the loss function.

    vf_coef: the coefficient for the value loss, which weights its contribution to the overall loss. Denoted by c_1 in the paper.
    '''
    critic_prediction = critic(mb_obs)

    return 0.5 * vf_coef * (critic_prediction - mb_returns).pow(2).mean()


def calc_entropy_loss(probs: Categorical, ent_coef: float):
    '''Return the entropy loss term.

    ent_coef: the coefficient for the entropy loss, which weights its contribution to the overall loss. Denoted by c_2 in the paper.
    '''
    return ent_coef * probs.entropy().mean()

# %%
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    t.manual_seed(seed)

gamma = 0.9
norvig = Norvig(-0.04)

pi_up = np.zeros(12, dtype=int)  # always go up
pi_caution = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 3, 3, 3], dtype=int)  # cautiously walk towards +1
pi_risky = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 3], dtype=int)  # shortest path to +1
pi_suicidal = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0], dtype=int)  # shortest path to +1 or -1
pi_immortal = np.array([2, 3, 3, 0, 1, 0, 2, 0, 2, 3, 3, 3], dtype=int)  # hide behind wall

policies = np.stack((pi_caution, pi_risky, pi_suicidal, pi_immortal, pi_up))
values = np.array(list(map(lambda pi: policy_eval_exact(norvig, pi, gamma), policies)))

def linear_schedule(current_step: int, start_e: float, end_e: float, exploration_fraction: float, total_timesteps: int) -> float:
    """Return the appropriate epsilon for the current step.
    Epsilon should be start_e at step 0 and decrease linearly to end_e at step (exploration_fraction * total_timesteps).
    """
    "SOLUTION"
    duration = exploration_fraction * total_timesteps
    slope = (end_e - start_e) / duration
    return max(slope * current_step + start_e, end_e)

def test_linear_schedule(fn_to_test):
    expected = t.tensor([linear_schedule(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500)
        for step in range(500)])
    actual = t.tensor([fn_to_test(step, start_e=1.0, end_e=0.05, exploration_fraction=0.5, total_timesteps=500) 
        for step in range(500)])
    assert expected.shape == actual.shape
    np.testing.assert_allclose(expected, actual)

def test_policy_eval(fn_to_test, exact=False):

    # try a handful of random policies

    for pi in policies:
        pi = np.random.randint(Norvig.num_actions, size=(Norvig.num_states,))
        if exact:
            expected = policy_eval_exact(Norvig, pi, gamma=0.9)
        else:
            expected = policy_eval_numerical(Norvig, pi, gamma=0.9, eps=1e-8)
        actual = fn_to_test(Norvig, pi, gamma=0.9)
        assert actual.shape == (Norvig.num_states,)
        t.testing.assert_close(t.tensor(expected), t.tensor(actual))

def test_policy_improvement(fn_to_test):
    for v in values:
        expected = policy_improvement(Norvig, v, gamma)
        actual = fn_to_test(Norvig, v, gamma)
        t.testing.assert_close(t.tensor(expected), t.tensor(actual))

def test_find_optimal_policy(fn_to_test):

    # can't easily compare policies directly
    # as optimal policy is not unique
    # compare value functions instead

    gamma = 0.99

    env_mild = Norvig(-0.02)
    env_painful = Norvig(-0.1)
    env_hell = Norvig(-10)
    env_heaven = Norvig(10)
    enviros = [env_mild, env_painful, env_hell, env_heaven]

    for i in range(4):
        expected_pi_opt = policies[i]
        actual_pi_opt = fn_to_test(enviros[i], gamma)
        # print("Expected Policy")
        # print(enviros[i].render(expected_pi_opt))  # maybe have it print the policy in a nice way?
        # print(enviros[i].render(actual_pi_opt))
        val1 = policy_eval_exact(Norvig, expected_pi_opt, gamma)
        val2 = policy_eval_exact(Norvig, actual_pi_opt, gamma)
        t.testing.assert_close(t.tensor(val1), t.tensor(val2))


def _random_experience(num_actions, observation_shape, num_environments):
    obs = np.random.randn(num_environments, *observation_shape)
    actions = np.random.randint(0, num_actions - 1, (num_environments,))
    rewards = np.random.randn(num_environments)
    dones = np.random.randint(0, 1, (num_environments,)).astype(bool)
    next_obs = np.random.randn(num_environments, *observation_shape)
    return (obs, actions, rewards, dones, next_obs)

def test_replay_buffer_single(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, seed=1, device=t.device("cpu")
):
    """If the buffer has a single experience, that experience should always be returned when sampling."""
    rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    exp = _random_experience(num_actions, observation_shape, num_environments)
    rb.add(*exp)
    for _ in range(10):
        actual = rb.sample(1, device)
        t.testing.assert_close(actual.observations, t.tensor(exp[0]))
        t.testing.assert_close(actual.actions, t.tensor(exp[1]))
        t.testing.assert_close(actual.rewards, t.tensor(exp[2]))
        t.testing.assert_close(actual.dones, t.tensor(exp[3]))
        t.testing.assert_close(actual.next_observations, t.tensor(exp[4]))

def test_replay_buffer_deterministic(
    cls, buffer_size=5, num_actions=2, observation_shape=(4,), num_environments=1, device=t.device("cpu")
):
    """The samples chosen should be deterministic, controlled by the given seed."""
    for seed in [67, 88]:
        rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        rb2: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
        for _ in range(5):
            exp = _random_experience(num_actions, observation_shape, num_environments)
            rb.add(*exp)
            rb2.add(*exp)

        # Sequence of samples should be identical (ensuring they use self.rng)
        for _ in range(10):
            actual = rb.sample(2, device)
            actual2 = rb2.sample(2, device)
            for v, v2 in zip(asdict(actual).values(), asdict(actual2).values()):
                t.testing.assert_close(v, v2)

def test_replay_buffer_wraparound(
    cls, buffer_size=4, num_actions=2, observation_shape=(1,), num_environments=1, seed=3, device=t.device("cpu")
):
    """When the maximum buffer size is reached, older entries should be overwritten."""
    rb: ReplayBuffer = cls(buffer_size, num_actions, observation_shape, num_environments, seed)
    for i in range(6):
        rb.add(
            np.array([[float(i)]]),
            np.array([i % 2]),
            np.array([-float(i)]),
            np.array([False]),
            np.array([[float(i) + 1]]),
        )
    # Should be [4, 5, 2, 3] in the observations buffer now
    unique_obs = rb.sample(1000, device).observations.flatten().unique()
    t.testing.assert_close(unique_obs, t.arange(2, 6, device=device).to(dtype=unique_obs.dtype))


def test_epsilon_greedy_policy(fn_to_test):

    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", 0, 0, False, "test_eps_greedy_policy") for _ in range(5)])

    num_observations = np.array(envs.single_observation_space.shape, dtype=int).prod()
    num_actions = envs.single_action_space.n
    q_network = QNetwork(num_observations, num_actions)
    obs = t.randn((envs.num_envs, *envs.single_observation_space.shape))
    greedy_action = epsilon_greedy_policy(envs, q_network, np.random.default_rng(0), obs, 0)

    def get_actions(epsilon, seed):
        set_seed(seed)
        soln_actions = epsilon_greedy_policy(
            envs, q_network, np.random.default_rng(seed), obs, epsilon
        )
        set_seed(seed)
        their_actions = fn_to_test(envs, q_network, np.random.default_rng(seed), obs, epsilon)
        return soln_actions, their_actions

    def are_both_greedy(soln_acts, their_acts):
        return np.array_equal(soln_acts, greedy_action) and np.array_equal(their_acts, greedy_action)

    both_actions = [get_actions(0.1, seed) for seed in range(20)]
    assert all([soln_actions.shape == their_actions.shape for (soln_actions, their_actions) in both_actions])

    both_greedy = [are_both_greedy(*get_actions(0.1, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.9

    both_greedy = [are_both_greedy(*get_actions(0.5, seed)) for seed in range(100)]
    assert np.mean(both_greedy) >= 0.5

    both_greedy = [are_both_greedy(*get_actions(1, seed)) for seed in range(1000)]
    assert np.mean(both_greedy) > 0 and np.mean(both_greedy) < 0.1


def make_env(env_id: str, seed: int, idx: int, capture_video: bool, run_name: str):
    """Return a function that returns an environment after setting up boilerplate."""
    
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        obs = env.reset(seed=seed) # env.seed(seed) ?
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    
    return thunk

def window_avg(arr: Arr, window: int):
    """
    Computes sliding window average
    """
    return np.convolve(arr, np.ones(window), mode="valid") / window

def cummean(arr: Arr):
    """
    Computes the cumulative mean
    """
    return np.cumsum(arr) / np.arange(1, len(arr) + 1)

# Taken from https://stackoverflow.com/questions/42869495/numpy-version-of-exponential-weighted-moving-average-equivalent-to-pandas-ewm
# See https://en.wikipedia.org/wiki/Moving_average#Exponential_moving_average
def ewma(arr : Arr, alpha : float):
    '''
    Returns the exponentially weighted moving average of x.
    Parameters:
    -----------
    x : array-like
    alpha : float {0 <= alpha <= 1}
    Returns:
    --------
    ewma: numpy array
          the exponentially weighted moving average
    '''
    # Coerce x to an array
    s = np.zeros_like(arr)
    s[0] = arr[0]
    for i in range(1,len(arr)):
        s[i] = alpha * arr[i] + (1-alpha)*s[i-1]
    return s


def sum_rewards(rewards : List[int], gamma : float = 1):
    """
    Computes the total discounted sum of rewards for an episode.
    By default, assume no discount
    Input:
        rewards [r1, r2, r3, ...] The rewards obtained during an episode
        gamma: Discount factor
    Output:
        The sum of discounted rewards 
        r1 + gamma*r2 + gamma^2 r3 + ...
    """
    total_reward = 0
    for r in rewards[:0:-1]: #reverse, excluding first
        total_reward += r
        total_reward *= gamma
    total_reward += rewards[0]
    return total_reward

def ppo_parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return PPOArgs(**vars(args))


def test_agent(Agent):
    envs = gym.vector.SyncVectorEnv([make_env("CartPole-v1", i, i, False, "test-run") for i in range(5)])
    agent = Agent(envs)
    assert sum(p.numel() for p in agent.critic.parameters()) == 4545
    assert sum(p.numel() for p in agent.actor.parameters()) == 4610
    for name, param in agent.named_parameters():
        if "bias" in name:
            t.testing.assert_close(param.pow(2).sum(), t.tensor(0.0))

def test_compute_advantages(fn_to_test, verbose=False):

    t_ = 4
    env_ = 6
    next_value = t.randn(1, env_)
    next_done = t.randint(0, 2, (env_,))
    rewards = t.randn(t_, env_)
    values = t.randn(t_, env_)
    dones = t.randn(t_, env_)
    device = t.device("cpu")
    gamma = 0.95
    gae_lambda = 0.95
    args = (next_value, next_done, rewards, values, dones, device, gamma, gae_lambda)
    
    actual = fn_to_test(*args)
    expected = compute_advantages(*args)

    if verbose:
        print("actual", actual)
        print("expected", expected)

    t.testing.assert_close(actual, expected)

def test_calc_policy_loss(fn_to_test, verbose=False):

    minibatch = 3
    num_actions = 4
    probs = Categorical(logits=t.randn((minibatch, num_actions)))
    mb_action = t.randint(0, num_actions, (minibatch,))
    mb_advantages = t.randn((minibatch,))
    mb_logprobs = t.randn((minibatch,))
    clip_coef = 0.01
    expected = calc_policy_loss(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)
    actual = fn_to_test(probs, mb_action, mb_advantages, mb_logprobs, clip_coef)

    if verbose:
        print("actual", actual)
        print("expected", expected)

    t.testing.assert_close(actual.pow(2), expected.pow(2))
    if actual * expected < 0:
        print("Warning: you have calculated the negative of the policy loss, suitable for gradient descent.")
    print("All tests in `test_calc_policy_loss` passed.")

def test_calc_value_function_loss(fn_to_test, verbose=False):
    critic = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
    mb_obs = t.randn(5, 3)
    mb_returns = t.randn(5, 4)
    vf_coef = 0.5
    with t.inference_mode():
        expected = calc_value_function_loss(critic, mb_obs, mb_returns, vf_coef)
        actual = fn_to_test(critic, mb_obs, mb_returns, vf_coef)
        if verbose:
            print("actual", actual)
            print("expected", expected)
    if (actual - expected).abs() < 1e-4:
        print("All tests in `test_calc_value_function_loss` passed!")
    elif (0.5*actual - expected).abs() < 1e-4:
        raise Exception("Your result was half the expected value. Did you forget to use a factor of 1/2 in the mean squared difference?")
    t.testing.assert_close(actual, expected)

def test_calc_entropy_loss(calc_entropy_loss, verbose=False):
    probs = Categorical(logits=t.randn((3, 4)))
    ent_coef = 0.5
    expected = ent_coef * probs.entropy().mean()
    actual = calc_entropy_loss(probs, ent_coef)
    if verbose:
        print("actual", actual)
        print("expected", expected)
    t.testing.assert_close(expected, actual)

def test_minibatch_indexes(minibatch_indexes):
    for n in range(5):
        frac, minibatch_size = np.random.randint(1, 8, size=(2,))
        batch_size = frac * minibatch_size
        indices = minibatch_indexes(batch_size, minibatch_size)
        assert isinstance(indices, list)
        assert isinstance(indices[0], np.ndarray)
        assert len(indices) == frac
        np.testing.assert_equal(np.sort(np.stack(indices).flatten()), np.arange(batch_size))
    print("All tests in `test_minibatch_indexes` passed.")