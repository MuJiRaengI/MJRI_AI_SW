import abc


class Agent(abc.ABC):
    def __init__(self, mode="random"):
        self.mode = mode  # ['random', 'train', 'test']

    @abc.abstractmethod
    def select_action(self, state):
        pass

    @abc.abstractmethod
    def step(self, state, action, reward, next_state, done):
        pass

    @abc.abstractmethod
    def learn(self, state, action, reward, next_state, done):
        pass
