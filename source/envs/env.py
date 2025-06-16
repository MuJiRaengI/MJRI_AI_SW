import abc


class Env(abc.ABC):
    def __init__(self):
        self.fps = 30
        self.mode = None
        self.save_dir = None
        self.log_dir = "logs"

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def _self_play(self):
        pass

    @abc.abstractmethod
    def _random_play(self):
        pass

    @abc.abstractmethod
    def _train(self):
        pass

    @abc.abstractmethod
    def _test(self):
        pass

    @abc.abstractmethod
    def key_info(self) -> str:
        """Return key information for manual control."""
        return "No key information provided."

    def play(self, save_dir, mode="random"):
        self.save_dir = save_dir
        self.mode = mode
        if mode == "self_play":
            return self._self_play()
        elif mode == "random_play":
            return self._random_play()
        elif mode == "train":
            return self._train()
        elif mode == "test":
            return self._test()
        else:
            raise ValueError(f"Unknown mode: {mode}")
