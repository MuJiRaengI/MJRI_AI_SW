import abc


class Env(abc.ABC):
    def __init__(self):
        self.fps = 30
        self.scale = 1
        self.mode = None
        self.save_dir = None
        self.training_queue = None
        self.render_queue = None
        self.log_dir = "logs"

    @abc.abstractmethod
    def _self_play(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _random_play(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _train(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def _test(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def key_info(self) -> str:
        """Return key information for manual control."""
        return "No key information provided."

    def play(self, save_dir, mode="random", queue=None, *args, **kwargs):
        self.save_dir = save_dir
        self.mode = mode
        if mode == "self_play":
            self.render_queue = queue
            return self._self_play(*args, **kwargs)
        elif mode == "random_play":
            self.render_queue = queue
            return self._random_play(*args, **kwargs)
        elif mode == "train":
            self.training_queue = queue
            # return self._train(*args, **kwargs)
            return self._train_bbf(*args, **kwargs)
        elif mode == "test":
            self.render_queue = queue
            # return self._test(*args, **kwargs)
            return self._test_bbf(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")
