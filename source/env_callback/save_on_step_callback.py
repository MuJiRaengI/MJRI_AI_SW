from stable_baselines3.common.callbacks import BaseCallback
import os
import time
import json


class SaveOnStepCallback(BaseCallback):
    """
    Custom callback for saving a model every `save_freq` steps, and logging progress/episode info to json and console.
    """

    def __init__(
        self,
        save_freq: int,
        logging_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        log_dir: str = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.logging_freq = logging_freq
        self.save_dir = save_path
        self.save_ckpt_dir = os.path.join(save_path, "ckpt")
        self.name_prefix = name_prefix
        self.log_dir = log_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.last_print = 0
        self.start_time = None
        os.makedirs(self.save_dir, exist_ok=True)
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        self.start_time = time.time()

        # 클래스 변수로 현재 에피소드 reward/length 누적용 변수 추가
        self._current_ep_reward = 0
        self._current_ep_length = 0

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
                self.save_dir, f"{self.name_prefix}_{self.n_calls}_steps"
            )
            self.model.save(model_path)
            print(f"Saved model checkpoint to {model_path}")

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
        return True
