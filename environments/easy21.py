from curses import intrflush
from dataclasses import dataclass, field
from enum import Enum
from os import stat
from typing import Tuple, List

import polars as pl
import numpy as np

from .base import Action, State, Policy, Environment, Agent, Transition
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


@dataclass
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

    def step_greedy(self, state: State, agent: Agent) -> EasyAction:

        # Evaluate Q function for both state, action pairs
        sa_stick, sa_hit = generate_state_action_pairs(state, EasyAction)
        Q_stick, Q_hit = agent.get_Q(sa_stick), agent.get_Q(sa_hit)

        # Greedy policy: take the one with the highest q value
        if Q_stick > Q_hit:
            return EasyAction.STICK
        elif Q_stick < Q_hit:
            return EasyAction.HIT
        else:  # Draw, select random action
            return np.random.choice([EasyAction.HIT, EasyAction.STICK])

    def step(self, state: State, agent: Agent) -> EasyAction:

        mode = self.sample_mode(agent.get_state_count(state))

        if mode is Mode.EXPLOIT:  # Exploit: take best action given q
            return self.step_greedy(state, agent)
        else:  # Explore: randomly decide between actions
            return np.random.choice([EasyAction.HIT, EasyAction.STICK])


## Agent


class Easy21Agent(Agent):
    def get_Q(self, sa_pair: StateActionPair = None):

        if sa_pair:
            return self.Q(sa_pair.key)

        df = pl.DataFrame({"key": list(self.Q.r.keys()), "q": list(self.Q.r.values())})

        df_Q = (
            df.with_columns(pl.col("key").str.split_exact("-", 3))
            .unnest("key")
            .rename(
                {
                    "field_0": "Dealer showing",
                    "field_1": "Player sum",
                    "field_2": "Terminal",
                    "field_3": "Action",
                }
            )
            .drop("Terminal")
            .with_columns(
                [
                    pl.col("Dealer showing").cast(pl.Int32),
                    pl.col("Player sum").cast(pl.Int32),
                ]
            )
            .sort(["Dealer showing", "Player sum"])
        )

        return df_Q

    def get_V(self):

        df_Q = self.get_Q()

        df_V = (
            df_Q.select(["Dealer showing", "Player sum", "q"])
            .groupby(by=["Dealer showing", "Player sum"])
            .max()
            .sort(["Dealer showing", "Player sum"])
            .pivot(values="q", index="Dealer showing", columns="Player sum")
        )

        return df_V


class DealerAgent(Easy21Agent):
    def __init__(self) -> None:
        self.policy = DealerPolicy()

    def step(self, state: EasyState) -> EasyAction:
        return self.policy.step(state, state.player_sum)

    def optimize(self, transitions: List[Transition]) -> None:
        pass

    def get_Q(self, sa_pair: StateActionPair = None):
        raise NotImplementedError

    def get_V(self):
        raise NotImplementedError


class MCAgent(Easy21Agent):
    def __init__(self, policy: Policy) -> None:
        self.policy = policy

        self.Ns = CountRegistry()
        self.Nsa = CountRegistry()
        self.Q = ValueRegistry()

    def get_state_count(self, state: State) -> int:
        # Used in policy (we pass self to step())
        return self.Ns(state.key)

    def get_return(self, idx: int, transitions: List[Transition]) -> float:
        return transitions[-1].reward

    def step(self, state: EasyState) -> Action:
        action = self.policy.step(state, self)
        return action

    def optimize(self, transitions: List[Transition]) -> None:

        for idx, transition in enumerate(transitions):

            # Get return and state-action pair for current transition
            G = self.get_return(idx, transitions)
            sa_pair = StateActionPair(state=transition.state, action=transition.action)

            # Update counters
            self.Ns.increment(sa_pair.state.key)
            self.Nsa.increment(sa_pair.key)

            # Update Q for the current state-action pair
            q_current = self.Q(sa_pair.key)
            q_new = q_current + 1 / self.Nsa(sa_pair.key) * (G - q_current)
            self.Q.store(sa_pair.key, q_new)


class TDAgent(Easy21Agent):
    def __init__(self, policy: Policy, lmbda: float, gamma: float = 1) -> None:
        self.policy = policy
        self.lmbda = lmbda  # Lambda
        self.gamma = gamma  # Default=1, no discounting in Easy21

        self.Ns = CountRegistry()
        self.Nsa = CountRegistry()
        self.Q = ValueRegistry()
        self.E = CountRegistry()

    def get_state_count(self, state: State) -> int:
        # Used in policy (we pass self to step())
        return self.Ns(state.key)

    def step(self, state: State) -> Action:

        action = self.policy.step(state, self)
        sa_pair = StateActionPair(state, action)

        self.Ns.increment(sa_pair.state.key)
        self.Nsa.increment(sa_pair.key)

        return action

    def reset_episode(self):
        self.E = CountRegistry()

    def optimize(self, transitions: List[Transition]) -> None:

        assert len(transitions) == 1, "Expected list of transitions with length 1"
        t = transitions[-1]

        # Sample next action given next state from policy
        next_action = self.policy.step(t.next_state, self)

        # Assemble state - action pairs
        sa_current = StateActionPair(t.state, t.action)
        sa_next = StateActionPair(t.next_state, next_action)

        # Calculate delta for updating Q(S, A)
        delta = t.reward + self.gamma * self.Q(sa_next) - self.Q(sa_current)

        # Increment eligibility trace
        self.E.increment(sa_current.key)

        for key, E_current in self.E.r.items():

            q_current = self.Q(key)

            # Update Q for current state pair in proportion to eligibility trace
            q_new = q_current + 1 / self.Nsa(key) * delta * E_current
            self.Q.store(key, q_new)

            # Decay eligibility trace
            self.E.r[key] *= self.gamma * self.lmbda

        # Increment next action counter since we did not call step directly
        self.Ns.increment(sa_next.state.key)
        self.Nsa.increment(sa_next.key)

        return next_action


## Environment


class EasyEnvironment(Environment):
    def __init__(self):
        self.dealer_sum = None
        self.state = None

        self.reset()
        self.dealer_policy = DealerPolicy()

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
                return self.get_state(set_terminal=True), -1

            else:

                # Game continues: reward 0 (intermediate step)
                return self.get_state(), 0

        else:  # Player sticks, dealer (environment) policy runs

            while (
                self.dealer_policy.step(self.get_state(), self.dealer_sum)
                is EasyAction.HIT
            ):

                card = Card()
                self.dealer_sum += card.value

                if self._is_bust(self.dealer_sum):

                    # Player wins: reward +1
                    return self.get_state(set_terminal=True), 1

            # Dealer won't draw more cards - determine reward
            r = np.sign(self.state.player_sum - self.dealer_sum)

            return self.get_state(set_terminal=True), r
