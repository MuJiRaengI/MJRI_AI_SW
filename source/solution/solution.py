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
        # Path 객체로 변환
        obj.root = Path(data.get("root", ""))
        obj.name = Path(data.get("name", ""))
        obj.json_name = Path(data.get("json_name", ""))
        obj.task = data.get("task", None)
        # 다른 멤버가 있다면 여기에 추가
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
