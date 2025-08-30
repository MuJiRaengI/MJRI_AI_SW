import abc
import logging
import os
import time
import numpy as np

import torch
import torch.nn as nn


class Agent(abc.ABC):
    def __init__(
        self,
        save_dir: str,
        logging_freq: int,
        detailed_logging_freq: int,
    ):
        self.logging_freq = logging_freq
        self.detailed_logging_freq = detailed_logging_freq
        self.device = "cpu"

        # 로깅 설정
        self.logger = self._setup_logger(save_dir, "log.txt")

        # 모델 저장 관련 속성
        self.model = None
        self.save_dir = save_dir
        self.model_name = "model"
        self.max_best_models = 5
        self.best_mean_reward = -float("inf")
        self.episode_rewards = []
        self.total_steps = 0
        self.start_time = None

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

    def setup_model_saving(
        self,
        save_dir: str,
        model_name: str = "model",
        max_best_models: int = 5,
    ):
        """모델 저장 설정 (Best 모델만 저장)"""
        self.save_dir = save_dir
        self.model_name = model_name
        self.max_best_models = max_best_models
        os.makedirs(save_dir, exist_ok=True)
        self.logger.info(f"Best 모델 저장 설정 완료: {save_dir}")
        self.logger.info(f"최대 {max_best_models}개의 best 모델 유지")

    def save_best_model(self, model: nn.Module, mean_reward: float, step: int = None):
        """Best 모델 저장 (성능 개선 시에만)"""
        if self.save_dir is None:
            return

        if mean_reward > self.best_mean_reward:
            before_reward = self.best_mean_reward
            self.best_mean_reward = mean_reward
            step = step or self.total_steps
            # 소수점을 언더바로 변경하여 확장자와 구분
            reward_str = f"{mean_reward:.3f}".replace(".", "_")
            best_path = os.path.join(
                self.save_dir,
                f"{self.model_name}_best_{step}_reward_{reward_str}.pth",
            )

            try:
                # model.save(best_path)
                torch.save(model.state_dict(), best_path)
                self.logger.info(f"[BEST 모델] 새로운 최고 성능! 저장: {best_path}")
                self.logger.info(
                    f"평균 보상: {mean_reward:.3f} (이전: {before_reward:.3f})"
                )
                self.logger.info(f"스텝: {step}")
                self._cleanup_best_models()
                return True
            except Exception as e:
                self.logger.error(f"[에러] Best 모델 저장 실패: {e}")
        return False

    def save_final_model(self, model, step: int = None):
        """최종 모델 저장"""
        if self.save_dir is None:
            return

        step = step or self.total_steps
        final_path = os.path.join(self.save_dir, f"{self.model_name}_final_{step}.pth")

        try:
            # model.save(final_path)
            torch.save(model.state_dict(), final_path)
            self.logger.info(f"[최종 모델] 저장: {final_path}")
            if self.start_time:
                total_time = time.time() - self.start_time
                self.logger.info(f"총 훈련 시간: {self._format_time(total_time)}")
            if self.episode_rewards:
                self.logger.info(
                    f"최종 평균 보상: {np.mean(self.episode_rewards[-100:]):.3f}"
                )
                self.logger.info(f"최고 평균 보상: {self.best_mean_reward:.3f}")
        except Exception as e:
            self.logger.error(f"[에러] 최종 모델 저장 실패: {e}")

    def update_episode_rewards(self, rewards):
        """에피소드 보상 업데이트"""
        if isinstance(rewards, (list, np.ndarray)):
            self.episode_rewards.extend(rewards)
        else:
            self.episode_rewards.append(rewards)

        # 최근 1000개만 유지 (메모리 절약)
        if len(self.episode_rewards) > 1000:
            self.episode_rewards = self.episode_rewards[-1000:]

    def get_mean_reward(self, last_n: int = 100):
        """최근 N개 에피소드의 평균 보상"""
        if len(self.episode_rewards) < last_n:
            return np.mean(self.episode_rewards) if self.episode_rewards else 0.0
        return np.mean(self.episode_rewards[-last_n:])

    def check_and_save_best(self, model, step: int = None, min_episodes: int = 10):
        """성능 체크 후 best 모델 저장 (편의 메서드)"""
        if len(self.episode_rewards) >= min_episodes:
            mean_reward = self.get_mean_reward()
            return self.save_best_model(model, mean_reward, step)
        return False

    def _cleanup_best_models(self):
        """오래된 best 모델 정리"""
        try:
            best_files = []
            for f in os.listdir(self.save_dir):
                if f.startswith(f"{self.model_name}_best_") and f.endswith(".pth"):
                    try:
                        # 소수점이 언더바로 변경된 파일명 파싱
                        reward_str = f.split("_reward_")[1].replace(".pth", "")
                        # 언더바를 소수점으로 되돌려서 float로 변환
                        reward_val = float(reward_str.replace("_", "."))
                        best_files.append((f, reward_val))
                    except:
                        continue

            if len(best_files) > self.max_best_models:
                best_files.sort(key=lambda x: x[1], reverse=True)
                files_to_delete = best_files[self.max_best_models :]
                for file_to_delete, _ in files_to_delete:
                    os.remove(os.path.join(self.save_dir, file_to_delete))
        except Exception as e:
            self.logger.error(f"[에러] Best 모델 정리 실패: {e}")

    @abc.abstractmethod
    def learn(self):
        pass

    @abc.abstractmethod
    def predict(self):
        pass
