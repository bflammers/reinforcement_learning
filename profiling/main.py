from copy import copy
from typing import List
from collections import deque

import numpy as np

from environments.easy21 import EasyEnvironment, MCAgent, EpsilonGreedyPolicy
from environments.base import Transition

n_episodes = 10000
n_checkpoints = 100

episodes_per_checkpoint = round(n_episodes / n_checkpoints)
rewards = deque(maxlen=episodes_per_checkpoint)

checkpoints = []

env = EasyEnvironment()

agent = MCAgent(policy=EpsilonGreedyPolicy(N_zero=100))

for i in range(n_episodes):

    # Checkpoint: print progress and store value function
    if i % episodes_per_checkpoint == 0:

        mean_reward = np.array(rewards).mean() if rewards else 0
        print(
            f"Episode: {i:>10}/{n_episodes} "
            + f"--- {i/n_episodes*100:>5.1f} "
            + f"--- mean reward: {mean_reward} "
        )

    # Reset environment and episode transitions
    env.reset()
    episode: List[Transition] = []

    s = env.get_state()

    while not s.terminal:

        # Take step, first agent using current state, then environment using action
        a = agent.step(s)
        next_s, R = env.step(a)

        # Store rewards and transitions
        rewards.append(R)
        episode.append(Transition(s, a, next_s, R))

        # Set state to next state
        s = copy(next_s)

    agent.optimize(episode)
