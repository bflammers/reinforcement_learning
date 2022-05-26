from environments.easy21 import EasyEnvironment, EpsilonGreedyPolicy, TDAgent
from environments.base import Transition


if __name__ == "__main__":

    from copy import copy

    n_episodes = 1000
    env = EasyEnvironment()

    agent = TDAgent(policy=EpsilonGreedyPolicy(N_zero=100), lmbda=0.9)

    for i in range(n_episodes):

        # Reset environment and agent
        env.reset()
        agent.reset_episode()

        s = env.get_state()
        a = agent.step(s)

        while not s.terminal:

            # Take step, first environment using action, then agent using next state
            next_s, R = env.step(a)

            # Optimize Q
            a = agent.optimize([Transition(s, a, next_s, R)])

            # Set state, action to next state, next action
            s = copy(next_s)
