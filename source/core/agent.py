import abc
import logging
import os
import json
from datetime import datetime


class Agent(abc.ABC):
    def __init__(self, config: dict):
        self.config = config
        self.now = datetime.now().strftime("%Yy%mm%dd_%Hh%Mm%Ss")
        self.save_dir = os.path.join(self.config["save_dir"], self.now)

    def create_dir(self):
        # set logger
        self.logger = self._setup_logger(self.save_dir, "log.txt")

        # save config
        os.makedirs(self.save_dir, exist_ok=True)
        config_path = os.path.join(self.save_dir, "config.json")
        self.save_json(config_path, self.config)

    def _setup_logger(self, save_dir: str, log_filename: str):
        """파일 핸들러와 콘솔 핸들러를 가진 로거 설정"""
        logger = logging.getLogger(f"Agent_{id(self)}")
        logger.setLevel(logging.INFO)

        # 기존 핸들러 제거 (중복 방지)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # 로그 디렉토리 생성
        os.makedirs(save_dir, exist_ok=True)

        # 1. 파일 핸들러 설정
        log_file_path = os.path.join(save_dir, log_filename)
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.INFO)

        # 2. 콘솔 핸들러 설정
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # 3. 포맷터 설정
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s - %(message)s",
            datefmt="%Yy-%mm-%dd %Hh:%Mm:%Ss",
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 4. 핸들러 등록
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.info(f"로거 초기화 완료 - 로그 파일: {log_file_path}")
        return logger

    def _format_time(self, total_seconds: float) -> str:
        """초를 일/시/분/초 형태로 변환"""
        days = int(total_seconds // 86400)
        hours = int((total_seconds % 86400) // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)

        time_str = []
        if days > 0:
            time_str.append(f"{days}일")
        if hours > 0:
            time_str.append(f"{hours}시간")
        if minutes > 0:
            time_str.append(f"{minutes}분")
        if seconds > 0 or not time_str:  # 최소한 초는 표시
            time_str.append(f"{seconds}초")

        return " ".join(time_str)

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass

    def load_json(self, path: str) -> dict:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def save_json(self, path: str, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
