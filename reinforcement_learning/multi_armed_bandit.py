#%%
import os
from typing import Optional, Union, Tuple
import gym
import gym.envs.registration
import gym.spaces
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

MAIN = __name__ == "__main__"
max_episode_steps = 1000
IS_CI = os.getenv("IS_CI")
N_RUNS = 200 if not IS_CI else 5
#%%
ObsType = int
ActType = int

class MultiArmedBandit(gym.Env):
    action_space: gym.spaces.Discrete
    observation_space: gym.spaces.Discrete
    num_arms: int
    stationary: bool
    arm_reward_means: np.ndarray
    arm_star: int

    def __init__(self, num_arms=10, stationary=True):
        super().__init__()
        self.num_arms = num_arms
        self.stationary = stationary
        self.observation_space = gym.spaces.Discrete(1)
        self.action_space = gym.spaces.Discrete(num_arms)
        self.reset()

    def step(self, arm: ActType) -> Tuple[ObsType, float, bool, dict]:
        '''
        Note: some documentation references a new style which has (termination, truncation) bools in place of the done bool.
        '''
        assert self.action_space.contains(arm)
        if not self.stationary:
            q_drift = self.np_random.normal(loc=0.0, scale=0.01, size=self.num_arms)
            self.arm_reward_means += q_drift
            self.best_arm = int(np.argmax(self.arm_reward_means))
        reward = self.np_random.normal(loc=self.arm_reward_means[arm], scale=1.0)
        obs = 0
        done = False
        info = dict(best_arm=self.best_arm)
        return (obs, reward, done, info)

    def reset(
        self, seed: Optional[int] = None, return_info=False, options=None
    ) -> Union[ObsType, Tuple[ObsType, dict]]:
        super().reset(seed=seed)
        if self.stationary:
            self.arm_reward_means = self.np_random.normal(loc=0.0, scale=1.0, size=self.num_arms)
        else:
            self.arm_reward_means = np.zeros(shape=[self.num_arms])
        self.best_arm = int(np.argmax(self.arm_reward_means))
        if return_info:
            return (0, dict())
        else:
            return 0

    def render(self, mode="human"):
        assert mode == "human", f"Mode {mode} not supported!"
        bandit_samples = []
        for arm in range(self.action_space.n):
            bandit_samples += [np.random.normal(loc=self.arm_reward_means[arm], scale=1.0, size=1000)]
        plt.violinplot(bandit_samples, showmeans=True)
        plt.xlabel("Bandit Arm")
        plt.ylabel("Reward Distribution")
        plt.show()
# %%
gym.envs.registration.register(
    id="ArmedBanditTestbed-v0",
    entry_point=MultiArmedBandit,
    max_episode_steps=max_episode_steps,
    nondeterministic=True,
    reward_threshold=1.0,
    kwargs={"num_arms": 10, "stationary": True},
)
if MAIN:
    env = gym.make("ArmedBanditTestbed-v0")
    print("Our env inside its wrappers looks like: ", env)
# %%
class Agent:
    '''Base class for agents in a multi-armed bandit environment'''

    rng: np.random.Generator

    def __init__(self, num_arms: int, seed: int):
        self.num_arms = num_arms
        self.reset(seed)

    def get_action(self) -> ActType:
        raise NotImplementedError()

    def observe(self, action: ActType, reward: float, info: dict) -> None:
        pass

    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

def run_episode(env: gym.Env, agent: Agent, seed: int):
    (rewards, was_best) = ([], [])
    env.reset(seed=seed)
    agent.reset(seed=seed)
    done = False
    while not done:
        arm = agent.get_action()
        (obs, reward, done, info) = env.step(arm)
        agent.observe(arm, reward, info)
        rewards.append(reward)
        was_best.append(1 if arm == info["best_arm"] else 0)
    rewards = np.array(rewards, dtype=float)
    was_best = np.array(was_best, dtype=int)
    return (rewards, was_best)

def test_agent(env: gym.Env, agent: Agent, n_runs=200):
    all_rewards = []
    all_was_bests = []
    for seed in tqdm(range(n_runs)):
        (rewards, corrects) = run_episode(env, agent, seed)
        all_rewards.append(rewards)
        all_was_bests.append(corrects)
    return (np.array(all_rewards), np.array(all_was_bests))

class RandomAgent(Agent):
    def get_action(self) -> ActType:
        return self.rng.integers(self.num_arms)

if MAIN:
   random_agent = RandomAgent(10, 0)
   all_rewards, all_was_bests = test_agent(env, random_agent)
   print(all_was_bests)
   print("Total reward", all_rewards.sum())
   print("% best arm", all_was_bests.sum()/len(all_was_bests)/len(all_was_bests[0]))

# %%
def plot_rewards(all_rewards: np.ndarray):
    (n_runs, n_steps) = all_rewards.shape
    (fig, ax) = plt.subplots(figsize=(15, 5))
    ax.plot(all_rewards.mean(axis=0), label="Mean over all runs")
    quantiles = np.quantile(all_rewards, [0.05, 0.95], axis=0)
    ax.fill_between(range(n_steps), quantiles[0], quantiles[1], alpha=0.5)
    ax.set(xlabel="Step", ylabel="Reward")
    ax.axhline(0, color="red", linewidth=1)
    fig.legend()
    return fig

class RewardAveragingAgent(Agent):
    estimated_rewards: np.ndarray
    num_actions: np.ndarray
    step: int

    def __init__(self, num_arms: int, seed: int, epsilon: float, optimism: float):
        super().__init__(num_arms, seed)
        self.epsilon = epsilon
        self.optimism = optimism
        self.estimated_rewards = np.full(self.num_arms, self.optimism, dtype='float32')
        self.num_actions = np.ones(self.num_arms)
        self.step = 1

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return self.rng.integers(self.num_arms)
        else:
            return np.argmax(self.estimated_rewards)

    def observe(self, action, reward, info):
        self.estimated_rewards[action] = self.estimated_rewards[action] + (reward - self.estimated_rewards[action]) / self.num_actions[action]
        self.num_actions[action] += 1
        self.step += 1

    def reset(self, seed: int):
        super().reset(seed)
        # Also reset estimated_rewards, step, number_actions? ?

if MAIN:
    stationary=True
    num_arms=10
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    regular_reward_averaging = RewardAveragingAgent(num_arms, 0, epsilon=0.1, optimism=0)
    (all_rewards, all_corrects) = test_agent(env, regular_reward_averaging, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    fig = plot_rewards(all_rewards)
    optimistic_reward_averaging = RewardAveragingAgent(num_arms, 0, epsilon=0.1, optimism=5)
    (all_rewards, all_corrects) = test_agent(env, optimistic_reward_averaging, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
# %%
class CheatyMcCheater(Agent):
    best_arm: int

    def __init__(self, num_arms: int, seed: int):
        super().__init__(num_arms, seed)
        self.best_arm = 0

    def get_action(self):
        return self.best_arm

    def observe(self, action, reward, info):
        self.best_arm = info['best_arm']

if MAIN:
    cheater = CheatyMcCheater(num_arms, 0)
    (all_rewards, all_corrects) = test_agent(env, cheater, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
# %%
class UCBActionSelection(Agent):
    estimated_rewards: np.ndarray
    num_actions: np.ndarray
    step: int

    def __init__(self, num_arms: int, seed: int, c: float, epsilon: float = 1e-7):
        super().__init__(num_arms, seed)
        self.c = c
        self.step = 1
        self.estimated_rewards = np.zeros(self.num_arms, dtype='float32')
        self.num_actions = np.zeros(self.num_arms)
        self.epsilon = epsilon

    def get_action(self):
        if self.step % 100 == 102:
            print("step", self.step)
            print(self.estimated_rewards)
            print(self.c * np.sqrt(np.log(self.step) / (self.num_actions + self.epsilon)))
            print()
        return np.argmax(self.estimated_rewards + self.c * np.sqrt(np.log(self.step) / (self.num_actions + self.epsilon)))

    def observe(self, action, reward, info):
        self.estimated_rewards[action] = self.estimated_rewards[action] + (reward - self.estimated_rewards[action]) / self.step
        self.num_actions[action] += 1
        self.step += 1

    def reset(self, seed: int):
        super().reset(seed)

if MAIN:
    env = gym.make("ArmedBanditTestbed-v0", num_arms=num_arms, stationary=stationary)
    ucb = UCBActionSelection(num_arms, 0, c=2.0)
    (all_rewards, all_corrects) = test_agent(env, ucb, n_runs=N_RUNS)
    print(f"Frequency of correct arm: {all_corrects.mean()}")
    print(f"Average reward: {all_rewards.mean()}")
    plot_rewards(all_rewards)
# %%
