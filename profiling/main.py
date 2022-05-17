
from environments.easy21 import (
    EasyEnvironment,
    MCAgent,
    EpsilonGreedyPolicy
)

episodes = 10000

rewards = []

env = EasyEnvironment()

agent = MCAgent(
    policy=EpsilonGreedyPolicy(N_zero=5)
)

for i in range(episodes):
    
    if i % (round(episodes / 20)) == 0:
        print(f"Episode: {i:>10}/{episodes} --- {i/episodes*100:>5.1f}%")
    
    env.reset()
    s = env.get_state()
    agent.reset_episode()
    
    while not s.terminal:

        # Take step, first agent using current state, then environment using action
        a = agent.step(s)
        s, G = env.step(a)
        
        rewards.append(G)
    
    agent.optimize(G)