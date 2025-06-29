from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Queue
import os
import time
import json
import re
import gc
import torch
import psutil


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback for saving a model when ep_rew_mean improves, and logging progress/episode info to json and console.
    """

    def __init__(
        self,
        save_freq: int,
        logging_freq: int,
        save_dir: str,
        name_prefix: str = "rl_model",
        log_dir: str = None,
        progress_queue: Queue = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.logging_freq = logging_freq
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.log_dir = log_dir
        self.max_log_length = 1000
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_print = 0
        self.start_time = None
        self.progress_queue = progress_queue
        self.best_mean_reward = -float("inf")  # 최고 평균 보상
        os.makedirs(self.save_dir, exist_ok=True)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self._current_ep_reward = 0
        self._current_ep_length = 0

    def extract_step(self, fname: str) -> int:
        match = re.search(r"_(\d+)_steps\.zip", fname)
        return int(match.group(1)) if match else -1

    def _on_step(self) -> bool:
        # step마다 reward 누적
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]
        for reward, done in zip(rewards, dones):
            self._current_ep_reward += float(reward)
            self._current_ep_length += 1
            if done:
                self.episode_rewards.append(self._current_ep_reward)
                self.episode_rewards = self.episode_rewards[-self.max_log_length :]
                self.episode_lengths.append(self._current_ep_length)
                self.episode_lengths = self.episode_lengths[-self.max_log_length :]
                self._current_ep_reward = 0
                self._current_ep_length = 0

        # ep_rew_mean이 갱신될 때만 저장
        if len(self.episode_rewards) >= 10:
            ep_rew_mean = sum(self.episode_rewards[-100:]) / min(
                len(self.episode_rewards), 100
            )
            if ep_rew_mean > self.best_mean_reward:
                self.best_mean_reward = ep_rew_mean
                gc.collect()
                model_path = os.path.join(
                    self.save_dir,
                    f"{self.name_prefix}_best_{int(self.num_timesteps)}_{int(ep_rew_mean)}.zip",
                )
                tmp_path = model_path.replace("zip", "tmp")
                with torch.no_grad():
                    self.model.save(tmp_path)
                try:
                    os.replace(tmp_path, model_path)
                except Exception as e:
                    print(f"[ERROR] 파일 교체 실패: {e}")
                print(
                    f"[BEST] New best model saved: {model_path} (ep_rew_mean={ep_rew_mean:.2f})"
                )

        # 기존 logging 코드 유지
        if self.n_calls % self.logging_freq == 0:
            gc.collect()
            elapsed = time.time() - self.start_time if self.start_time else 0
            progress = (
                self.num_timesteps / self.model._total_timesteps
                if hasattr(self.model, "_total_timesteps")
                and self.model._total_timesteps
                else 0
            )
            log_data = {
                "episode_rewards": self.episode_rewards,
                "episode_lengths": self.episode_lengths,
                "timestep": int(self.num_timesteps),
                "progress": float(progress),
                "elapsed_seconds": float(elapsed),
            }
            log_path = (
                os.path.join(self.log_dir, f"{self.name_prefix}_train_log.json")
                if self.log_dir
                else None
            )
            if log_path:
                with open(log_path, "w", encoding="utf-8") as f:
                    json.dump(log_data, f, ensure_ascii=False, indent=2)
            if self.verbose > 0:
                print(
                    f"[Train] Step: {self.num_timesteps} | Episodes: {len(self.episode_rewards)} | Last reward: {self.episode_rewards[-1] if self.episode_rewards else 0} | Elapsed: {elapsed:.1f}s | Progress: {progress*100:.1f}%"
                )
                print(f"중간 로그가 저장되었습니다: {log_path}")

            if self.progress_queue is not None:
                self.progress_queue.put(("progress", self.num_timesteps))
        return True
