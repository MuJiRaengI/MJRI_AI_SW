import os
import sys

sys.path.append(os.path.abspath("."))

import random
import time
import gym
import pygame
import keyboard
import PySide6.QtWidgets as QtWidgets
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from source.envs.env import Env
import json


class CartPole(Env):
    def __init__(self):
        super().__init__()
        self.obs, self.info = None, None
        self.done = False

    def reset(self):
        pygame.init()
        self.env = gym.make("CartPole-v1", render_mode="human")
        self.obs, self.info = self.env.reset()
        self.done = False

    def _self_play(self):
        self.reset()
        running = True
        action = None  # None이면 랜덤, 0: left, 1: right
        clock = pygame.time.Clock()
        pygame.display.set_caption(
            "CartPole-v1 Controller (A: Left, D: Right, ESC: Quit)"
        )
        QtWidgets.QMessageBox.information(
            None,
            "CartPole 수동 조작 안내",
            "A: 왼쪽, D: 오른쪽, ESC: 종료 (아무 키도 안누르면 랜덤)",
        )
        while running:
            if keyboard.is_pressed("a"):
                action = 0  # left
            elif keyboard.is_pressed("d"):
                action = 1  # right
            else:
                action = None  # 아무키도 안누르면 랜덤
            if action is None:
                action = random.choice([0, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                obs, info = self.env.reset()
            if keyboard.is_pressed("esc"):
                running = False
            clock.tick(getattr(self, "fps", 30))
        self.env.close()
        pygame.quit()

    def _random_play(self):
        self.reset()
        pygame.display.set_caption("CartPole-v1 Random Play (ESC: Quit)")
        clock = pygame.time.Clock()
        running = True
        while running:
            if keyboard.is_pressed("esc"):
                running = False
                continue
            action = random.choice([0, 1])
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                obs, info = self.env.reset()
            clock.tick(getattr(self, "fps", 30))
        self.env.close()
        pygame.quit()

    def _train(self):
        self.reset()
        log_dir = os.path.join(self.save_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ppo_cartpole_train_log.json")
        save_path = os.path.join(self.save_dir, "ppo_cartpole.zip")

        model = PPO("MlpPolicy", self.env, verbose=1)

        episode_rewards = []
        episode_lengths = []
        total_timesteps = 100000
        timestep = 0
        last_print = 0
        start_time = time.time()
        obs, info = self.env.reset()
        while timestep < total_timesteps:
            done = False
            ep_reward = 0
            ep_length = 0
            while not done and timestep < total_timesteps:
                action, _ = model.predict(obs, deterministic=False)
                step_result = self.env.step(action)
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                else:
                    obs, reward, done, info = step_result
                    terminated = done
                    truncated = False
                self.env.render()
                ep_reward += reward[0] if isinstance(reward, (list, tuple)) else reward
                ep_length += 1
                done = (
                    terminated[0] or truncated[0]
                    if isinstance(terminated, (list, tuple))
                    else terminated or truncated
                )
                timestep += 1
                if timestep % 1000 == 0:
                    # 중간 저장
                    model.save(save_path)
                    progress = timestep / total_timesteps
                    elapsed = time.time() - start_time
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
                if timestep - last_print >= 1000:
                    elapsed = time.time() - start_time
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
        elapsed = time.time() - start_time
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

    def _test(self):
        model_path = os.path.join(self.save_dir, "ppo_cartpole.zip")
        if not os.path.exists(model_path):
            print(f"모델 파일이 존재하지 않습니다: {model_path}")
            return
        model = PPO.load(model_path)
        self.reset()  # self.env, self.obs, self.info 초기화
        obs, info = self.env.reset()
        while True:
            if keyboard.is_pressed("esc"):
                break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.env.render()
            if terminated or truncated:
                obs, info = self.env.reset()
            time.sleep(1 / getattr(self, "fps", 30))
        self.env.close()
