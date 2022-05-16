import abc
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple
from copy import copy


class Action(Enum):
    pass


@dataclass
class State:
    pass

    @property
    def key(self) -> str:
        raise NotImplementedError


## Environment


class Environment(abc.ABC):

    def reset(self) -> None:
        raise NotImplementedError

    def get_state(self, set_terminal=True) -> State:
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
    def step(self, state: State) -> Action:
        pass
