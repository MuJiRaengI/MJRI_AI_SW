import json


class Solution:
    def __init__(self):
        # solution information
        self.save_path = None
        self.solution_name = None

        # task information
        self.task = None

    def to_dict(self):
        return self.__dict__

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        obj.__dict__.update(data)
        return obj

    def save_json(self, file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def load_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
