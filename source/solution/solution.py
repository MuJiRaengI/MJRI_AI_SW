import json
from pathlib import Path


class Solution:
    def __init__(self):
        # solution information
        self.root = Path()
        self.name = Path()
        self.json_name = Path()

        # task information
        self.task = None

        self.target_window = None
        self.screen_x = 0
        self.screen_y = 0
        self.screen_w = 0
        self.screen_h = 0

    def to_dict(self):
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                result[k] = str(v)
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, data):
        obj = cls()
        for k, v in data.items():
            # Path 타입 멤버 자동 변환 (필요시)
            if hasattr(obj, k):
                if isinstance(getattr(obj, k), Path):
                    setattr(obj, k, Path(v))
                else:
                    setattr(obj, k, v)
            else:
                setattr(obj, k, v)
        return obj

    def save_json(self, file_path=None):
        if file_path is None:
            file_path = self.root / self.name / self.json_name
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=4)

    @classmethod
    def load_json(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
