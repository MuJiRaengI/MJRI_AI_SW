import abc
import json


class EnvRunner(abc.ABC):
    def __init__(self):
        self.mode = None
        self.training_queue = None
        self.render_queue = None
        self.config_path = None
        self.running = False

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

    def play(self, config_path: str, mode: str = "random", queue=None, *args, **kwargs):
        self.config_path = config_path
        self.mode = mode
        if mode == "self_play":
            self.render_queue = queue
            return self._self_play(*args, **kwargs)
        elif mode == "random_play":
            self.render_queue = queue
            return self._random_play(*args, **kwargs)
        elif mode == "train":
            self.training_queue = queue
            return self._train(*args, **kwargs)
        elif mode == "test":
            self.render_queue = queue
            return self._test(*args, **kwargs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def load_json(self, path: str) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
