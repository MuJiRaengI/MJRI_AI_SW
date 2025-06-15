import os
import sys
import time as pytime

sys.path.append(os.path.abspath("."))

import gym
import keyboard
import glob
import json

import PySide6.QtWidgets as QtWidgets

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3.common.callbacks import CheckpointCallback

from stable_baselines3.common.callbacks import BaseCallback

from source.envs.env import Env


class Breakout(Env):
    def __init__(self):
        super().__init__()
        self.obs, self.info = None, None
        self.done = False
        self.fps = 60

    def reset(self):
        # self.env = gym.make("ALE/Breakout-v5", render_mode="human")
        self.env = gym.make("Breakout-v0", render_mode="human")
        self.obs, self.info = self.env.reset()
        self.done = False

    def _self_play(self):
        self.reset()
        running = True
        action = 0
        QtWidgets.QMessageBox.information(
            None,
            "Breakout 수동 조작 안내",
            "A: 왼쪽, D: 오른쪽, SPACE: 발사, ESC: 종료",
        )
        while running:
            # 키 입력 감지
            if keyboard.is_pressed("a"):
                action = 3  # LEFT
            elif keyboard.is_pressed("d"):
                action = 2  # RIGHT
            elif keyboard.is_pressed("space"):
                action = 1  # FIRE
            else:
                action = 0  # NOOP

            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                self.env.reset()

            if keyboard.is_pressed("esc"):
                running = False

            pytime.sleep(1 / self.fps)

        self.env.close()

    def _random_play(self):
        self.reset()
        running = True
        while running:
            if keyboard.is_pressed("esc"):
                running = False
                continue
            action = self.env.action_space.sample()
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                self.env.reset()
            pytime.sleep(1 / getattr(self, "fps", 30))
        self.env.close()

    def _train(self):
        self.reset()
        log_dir = os.path.join(self.save_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ppo_breakout_train_log.json")
        save_path = os.path.join(self.save_dir, "ppo_breakout.zip")

        model = PPO("CnnPolicy", self.env, verbose=1)
        episode_rewards = []
        episode_lengths = []
        obs, info = self.env.reset()
        total_timesteps = 1000000
        timestep = 0
        last_print = 0
        start_time = pytime.time()
        while timestep < total_timesteps:
            done = False
            ep_reward = 0
            ep_length = 0
            while not done and timestep < total_timesteps:
                action, _ = model.predict(obs, deterministic=False)
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.env.render()
                ep_reward += reward
                ep_length += 1
                done = terminated or truncated
                timestep += 1
                if timestep % 1000 == 0:
                    # 중간 저장
                    model.save(save_path)
                    progress = timestep / total_timesteps
                    elapsed = pytime.time() - start_time
                    safe_episode_rewards = [float(r) for r in episode_rewards]
                    safe_episode_lengths = [int(l) for l in episode_lengths]
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "episode_rewards": safe_episode_rewards,
                                "episode_lengths": safe_episode_lengths,
                                "timestep": int(timestep),
                                "progress": float(progress),
                                "elapsed_seconds": float(elapsed),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"중간 로그가 저장되었습니다: {log_path}")
                # 진행상황 출력 (1000스텝마다)
                if timestep - last_print >= 1000:
                    elapsed = pytime.time() - start_time
                    percent = 100 * timestep / total_timesteps
                    print(
                        f"[Train] Step: {timestep}/{total_timesteps} | Episodes: {len(episode_rewards)} | Last reward: {ep_reward} | Elapsed: {elapsed:.1f}s | Progress: {percent:.1f}%"
                    )
                    last_print = timestep
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            obs, info = self.env.reset()
        # 마지막 저장
        model.save(save_path)
        progress = timestep / total_timesteps
        elapsed = pytime.time() - start_time
        safe_episode_rewards = [float(r) for r in episode_rewards]
        safe_episode_lengths = [int(l) for l in episode_lengths]
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "episode_rewards": safe_episode_rewards,
                    "episode_lengths": safe_episode_lengths,
                    "timestep": int(timestep),
                    "progress": float(progress),
                    "elapsed_seconds": float(elapsed),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"모델이 저장되었습니다: {save_path}")
        print(f"학습 로그가 저장되었습니다: {log_path}")
        self.env.close()

    def _test(self, model_path=None):
        if model_path is None:
            ckpts = sorted(
                glob.glob(os.path.join(self.save_dir, "ppo_breakout_checkpoint_*.zip")),
                reverse=True,
            )
            if ckpts:
                model_path = ckpts[0]
            else:
                model_path = os.path.join(self.save_dir, "ppo_breakout.zip")
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            return
        print(f"불러온 모델: {model_path}")
        model = PPO.load(model_path)
        self.reset()
        obs, info = self.env.reset()
        while True:
            if keyboard.is_pressed("esc"):
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                obs, info = self.env.reset()
            pytime.sleep(1 / getattr(self, "fps", 30))
        self.env.close()
