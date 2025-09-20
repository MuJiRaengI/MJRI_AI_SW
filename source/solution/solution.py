import json
import os


class Solution:
    def __init__(self):
        # solution information
        self.root = ""
        self.name = ""
        self.json_name = ""

        self.target_window = None
        self.screen_x = 0
        self.screen_y = 0
        self.screen_w = 100
        self.screen_h = 100

        self.game_config = None

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            result[k] = v
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        for k, v in data.items():
            setattr(obj, k, v)
        return obj

    def save_json(self, file_path=None):
        if file_path is None:
            file_path = os.path.join(self.root, self.name, self.json_name)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def load_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
