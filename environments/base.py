import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, List
from copy import copy
from collections import namedtuple


class Action(Enum):
    pass


@dataclass
class State:
    pass

    @property
    def key(self) -> str:
        raise NotImplementedError


# From https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


## Environment


class Environment(abc.ABC):

    def reset(self) -> None:
        raise NotImplementedError

    def get_state(self, set_terminal=False) -> State:
        state = copy(self.state)
        state.terminal = set_terminal
        return state

    @abc.abstractmethod
    def step(self, action: Action) -> Tuple[State, float]:
        pass


## Agent


class Policy(abc.ABC):
    @abc.abstractmethod
    def step(self, state: State) -> Action:
        pass


class Agent(abc.ABC):

    def optimize(self):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, transitions: List[Transition]) -> Action:
        pass
