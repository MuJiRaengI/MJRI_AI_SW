from stable_baselines3.common.callbacks import BaseCallback
from multiprocessing import Queue
import os
import time
import json
import re


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback for saving a model every `save_freq` steps, and logging progress/episode info to json and console.
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
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_print = 0
        self.start_time = None
        self.progress_queue = progress_queue
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
                self.episode_lengths.append(self._current_ep_length)
                self._current_ep_reward = 0
                self._current_ep_length = 0

        # 중간 세이브 및 로그
        if self.n_calls % self.save_freq == 0:
            model_path = os.path.join(
                self.save_dir, f"{self.name_prefix}_{self.n_calls}_steps.zip"
            )

            tmp_path = model_path.replace("zip", "tmp")
            self.model.save(tmp_path)
            os.replace(tmp_path, model_path)
            print(f"Saved model checkpoint to {model_path}")

            # 최근 5개만 남기고 이전 체크포인트 삭제
            checkpoints = [
                fname
                for fname in os.listdir(self.save_dir)
                if fname.startswith(self.name_prefix)
                and fname.endswith("_steps.zip")
                and re.search(r"_(\d+)_steps\.zip", fname)
            ]
            checkpoints.sort(key=self.extract_step)
            print(checkpoints)
            for old_ckpt in checkpoints[:-5]:
                try:
                    os.remove(os.path.join(self.save_dir, old_ckpt))
                    print(f"Removed old checkpoint: {old_ckpt}")
                except Exception as e:
                    print(f"Could not remove old checkpoint {old_ckpt}: {e}")

        if self.n_calls % self.logging_freq == 0:
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
