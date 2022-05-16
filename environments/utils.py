from .base import State, Action


class StateActionPair:
    def __init__(self, state: State, action: Action):
        self.state = state
        self.action = action

    @property
    def key(self):
        return f"{self.state.key}-{self.action.name}"


def generate_state_action_pairs(state: State, actions: Action):
    return (StateActionPair(state, action) for action in actions)


class Registry:
    def __init__(self):
        self.r = dict()

    def __call__(self, key):
        return self.r.get(key, 0)


class CountRegistry(Registry):
    def increment(self, key):
        current = self.r.get(key, 0)
        self.r[key] = current + 1


class ValueRegistry(Registry):
    def store(self, key, value):
        self.r[key] = value
