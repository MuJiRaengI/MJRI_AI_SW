import os
import time

import torch
import torch.nn as nn
import numpy as np

from source.core import Agent


class ReinforcementLearning(Agent):
    def __init__(self, config: dict):
        super().__init__(config)

        self.model = None
        self.model_name = "model"
        self.save_ckpt_dir = "best_models"
        self.max_best_models = 5
        self.best_mean_reward = -float("inf")
        self.episode_rewards = []
        self.total_steps = 0
        self.start_time = None
        self.save_best_interval = 100
        self.detailed_logging_freq = 100

    def make_env(self):
        raise NotImplementedError("make_env method must be implemented in subclass")

    def make_env_vector(self):
        raise NotImplementedError(
            "make_env_vector method must be implemented in subclass"
        )

    def learn(self):
        raise NotImplementedError("make_env method must be implemented in subclass")

    def predict(self, obs):
        raise NotImplementedError("predict method must be implemented in subclass")

    def setup_model_saving(
        self,
        save_dir: str,
        model_name: str = "model",
        max_best_models: int = 5,
    ):
        """모델 저장 설정 (Best 모델만 저장)"""
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
                self.save_ckpt_dir,
                f"{self.model_name}_best_{step}_reward_{reward_str}.pth",
            )

            try:
                # model.save(best_path)
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
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
            save_dir = os.path.join(self.save_dir, self.save_ckpt_dir)
            for f in os.listdir(save_dir):
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
                    os.remove(os.path.join(save_dir, file_to_delete))
        except Exception as e:
            self.logger.error(f"[에러] Best 모델 정리 실패: {e}")
