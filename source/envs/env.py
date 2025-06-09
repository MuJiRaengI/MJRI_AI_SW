import abc


class Env(abc.ABC):
    def __init__(self):
        self.mode = None

    @abc.abstractmethod
    def _self_play(self):
        pass

    @abc.abstractmethod
    def _random_paly(self):
        pass

    @abc.abstractmethod
    def _train(self):
        pass

    @abc.abstractmethod
    def _test(self):
        pass

    def play(self, mode="random"):
        self.mode = mode
        if mode == "self_play":
            return self._self_play()
        elif mode == "random_play":
            return self._random_paly()
        elif mode == "train":
            return self._train()
        elif mode == "test":
            return self._test()
        else:
            raise ValueError(f"Unknown mode: {mode}")
