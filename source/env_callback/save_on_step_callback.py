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
    @staticmethod
    def get_latest_last_model_path(save_dir, name_prefix):
        """가장 최근(last) 모델 파일 경로 반환"""
        import re

        last_files = [
            fname
            for fname in os.listdir(save_dir)
            if fname.startswith(f"{name_prefix}_last_") and fname.endswith(".zip")
        ]
        if not last_files:
            return None

        def extract_step(fname):
            m = re.search(r"_last_(\\d+)\\.zip", fname)
            return int(m.group(1)) if m else -1

        last_files.sort(key=extract_step, reverse=True)
        return os.path.join(save_dir, last_files[0])

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
        max_best_models: int = 5,  # 최대 저장할 best 모델 개수
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.logging_freq = logging_freq
        self.save_dir = save_dir
        self.name_prefix = name_prefix
        self.log_dir = log_dir
        self.max_log_length = 1000
        self.max_best_models = max_best_models  # 최대 저장할 best 모델 개수
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

    def extract_reward(self, fname: str) -> float:
        """파일명에서 보상 값을 추출 (음수, 소수점 포함)"""
        match = re.search(r"_best_\d+_(-?\d+\.\d+)\.zip", fname)
        return float(match.group(1)) if match else None

    def cleanup_old_best_models(self):
        """가장 성능이 낮은 best 모델들을 삭제하여 최대 개수 유지"""
        try:
            # best 모델 파일들 찾기
            best_files = []
            for fname in os.listdir(self.save_dir):
                if fname.startswith(f"{self.name_prefix}_best_") and fname.endswith(
                    ".zip"
                ):
                    reward = self.extract_reward(fname)
                    if reward is not None:  # None이 아닌 모든 보상 값 포함 (음수 포함)
                        best_files.append((fname, reward))

            # print(
            #     f"[CLEANUP] 발견된 best 모델 {len(best_files)}개: {[f'{fname} (reward={reward})' for fname, reward in best_files]}"
            # )

            # 보상 기준으로 정렬 (높은 순)
            best_files.sort(key=lambda x: x[1], reverse=True)
            # print(
            #     f"[CLEANUP] 정렬 후: {[f'{fname} (reward={reward})' for fname, reward in best_files]}"
            # )

            # 최대 개수를 초과하는 파일들 삭제
            if len(best_files) > self.max_best_models:
                files_to_keep = best_files[: self.max_best_models]
                files_to_delete = best_files[self.max_best_models :]
                # print(
                #     f"[CLEANUP] 유지할 모델 {len(files_to_keep)}개: {[fname for fname, _ in files_to_keep]}"
                # )
                # print(
                #     f"[CLEANUP] 삭제할 모델 {len(files_to_delete)}개: {[fname for fname, _ in files_to_delete]}"
                # )

                for fname, reward in files_to_delete:
                    file_path = os.path.join(self.save_dir, fname)
                    try:
                        os.remove(file_path)
                        # print(
                        #     f"[CLEANUP] 이전 best 모델 삭제: {fname} (reward={reward})"
                        # )
                    except Exception as e:
                        print(f"[ERROR] 파일 삭제 실패 {fname}: {e}")
            else:
                pass
                # print(f"[CLEANUP] 현재 모델 개수 {len(best_files)}개로 정리 불필요")
        except Exception as e:
            print(f"[ERROR] best 모델 정리 실패: {e}")

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

        # best 모델 저장 (ep_rew_mean이 갱신될 때만)
        if len(self.episode_rewards) >= 10:
            ep_rew_mean = sum(self.episode_rewards[-100:]) / min(
                len(self.episode_rewards), 100
            )
            if ep_rew_mean > self.best_mean_reward:
                self.best_mean_reward = ep_rew_mean
                gc.collect()
                # 소수점 3자리까지 파일명에 저장
                model_path = os.path.join(
                    self.save_dir,
                    f"{self.name_prefix}_best_{int(self.num_timesteps)}_{ep_rew_mean:.3f}.zip",
                )
                tmp_path = model_path.replace("zip", "tmp")
                with torch.no_grad():
                    self.model.save(tmp_path)
                try:
                    os.replace(tmp_path, model_path)
                except Exception as e:
                    print(f"[ERROR] 파일 교체 실패: {e}")
                print(
                    f"[BEST] New best model saved: {model_path} (ep_rew_mean={ep_rew_mean:.3f})"
                )

                # 이전 best 모델들 정리 (상위 5개만 유지)
                self.cleanup_old_best_models()

        # last 모델은 save_freq마다 저장 (항상 최신 1개만 유지)
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            gc.collect()
            # 기존 last 모델 모두 삭제
            while True:
                last_path = self.get_latest_last_model_path(
                    self.save_dir, self.name_prefix
                )
                if last_path is None:
                    break
                try:
                    os.remove(last_path)
                except Exception as e:
                    print(f"[ERROR] 기존 last 모델 삭제 실패: {e}")
                    break
            # 새 last 모델 저장
            last_model_path = os.path.join(
                self.save_dir,
                f"{self.name_prefix}_last_{int(self.num_timesteps)}.zip",
            )
            last_tmp_path = last_model_path.replace("zip", "tmp")
            try:
                with torch.no_grad():
                    self.model.save(last_tmp_path)
                os.replace(last_tmp_path, last_model_path)
                if self.verbose > 0:
                    print(f"[LAST] Last model saved: {last_model_path}")
            except Exception as e:
                print(f"[ERROR] last 모델 저장 실패: {e}")

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
