from dataclasses import dataclass, field
from enum import Enum
from os import stat
from typing import Tuple, List


import numpy as np

from .base import Action, State, Policy, Environment, Agent
from .utils import (
    CountRegistry,
    ValueRegistry,
    StateActionPair,
    generate_state_action_pairs,
)


## CARDS


class CardColor(Enum):
    RED = 0
    BLACK = 1


@dataclass
class Card:
    number: int = field(default_factory=lambda: np.random.randint(1, 11))
    color: CardColor = field(
        default_factory=lambda: np.random.choice(
            [CardColor.RED, CardColor.BLACK], p=[1 / 3, 2 / 3]
        )
    )

    @property
    def value(self):
        if self.color is CardColor.BLACK:
            return self.number
        else:
            return -self.number


## Action


class EasyAction(Action):
    STICK = 0
    HIT = 1


## State


class EasyState(State):
    dealer_showing: int = field(
        default_factory=lambda: Card(color=CardColor.BLACK).value
    )
    player_sum: int = field(default_factory=lambda: Card(color=CardColor.BLACK).value)
    terminal: bool = False

    @property
    def key(self):
        return f"{self.dealer_showing}-{self.player_sum}-{self.terminal}"


## Policy


class Mode(Enum):
    EXPLOIT = 0
    EXPLORE = 1


class DealerPolicy(Policy):
    def step(self, state: State, current_sum: int) -> EasyAction:

        if current_sum < 17:
            return EasyAction.HIT

        return EasyAction.STICK


class EpsilonGreedyPolicy(Policy):
    def __init__(self, N_zero) -> None:
        self.N_zero = N_zero

    def sample_mode(self, state_count: int) -> Mode:
        # Randomly decide wether to exploit or explore
        eps = self.N_zero / (self.N_zero + state_count)
        mode = np.random.choice([Mode.EXPLOIT, Mode.EXPLORE], p=[1 - eps, eps])
        return mode

    def step(self, state: State, agent: Agent) -> EasyAction:

        mode = self.sample_mode(agent.get_state_count(state))

        # Exploit: take best action given q
        if mode is Mode.EXPLOIT:

            # Evaluate Q function for both state, action pairs
            sa_stick, sa_hit = generate_state_action_pairs(state, EasyAction)
            Q_stick, Q_hit = agent.get_q(sa_stick), agent.get_q(sa_hit)

            # Greedy policy: take the one with the highest q value
            if Q_stick > Q_hit:
                return EasyAction.STICK
            elif Q_stick < Q_hit:
                return EasyAction.HIT

        # Explore or draw in Q values: randomly decide between actions
        return np.random.choice([EasyAction.HIT, EasyAction.STICK])


## Agent


class DealerAgent(Agent):
    def __init__(self) -> None:
        self.policy = DealerPolicy()

    def step(self, state: EasyState) -> EasyAction:
        return self.policy.step(state, state.player_sum)

    def optimize(self, reward: float) -> None:
        pass


class MCAgent(Agent):
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

        self.Ns = CountRegistry()
        self.Nsa = CountRegistry()
        self.Q = ValueRegistry()

        self.episode: List[StateActionPair] = []

    def reset_episode(self):
        self.episode = []

    def get_state_count(self, state: State) -> int:
        return self.Ns(state.key)

    def get_q(self, sa_pair: StateActionPair = None):

        if sa_pair:
            return self.Q(sa_pair.key)

        return self.Q.r

    def step(self, state: EasyState) -> Action:
        action = self.policy.step(state, self)
        self.episode.append(StateActionPair(state, action))
        return action

    def optimize(self, G: float) -> None:

        for sa_pair in self.episode:

            # Update counters
            self.Ns.increment(sa_pair.state.key)
            self.Nsa.increment(sa_pair.key)

            # Update Q for the current state-action pair
            q_current = self.Q(sa_pair.key)
            q_new = q_current + 1 / self.Nsa(sa_pair.key) * (G - q_current)
            self.Q.store(sa_pair.key, q_new)


## Environment


class EasyEnvironment(Environment):
    def __init__(self):
        self.dealer_sum = None
        self.state = None

        self.reset()
        self.dealer = DealerAgent()

    def reset(self):
        self.state = EasyState()
        self.dealer_sum = self.state.dealer_showing

    @staticmethod
    def _is_bust(value: int) -> bool:
        if value > 21 or value < 1:
            return True

        return False

    def step(self, action: EasyAction) -> Tuple[State, float]:

        if action is EasyAction.HIT:
            card = Card()
            self.state.player_sum += card.value

            if self._is_bust(self.state.player_sum):

                # Player loses: reward -1
                return self.get_state(), -1

            else:

                # Game continues: reward 0 (intermediate step)
                return self.get_state(set_terminal=False), 0

        else:  # Player sticks, dealer (environment) policy runs

            while self.dealer.step(self.get_state(), self.dealer_sum) is EasyAction.HIT:

                card = Card()
                self.dealer_sum += card.value

                if self._is_bust(self.dealer_sum):

                    # Player wins: reward +1
                    return self.get_state(), 1

            # Dealer won't draw more cards - determine reward
            r = np.sign(self.state.player_sum - self.dealer_sum)

            return self.get_state(), r