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
from stable_baselines3.common.callbacks import CheckpointCallback
from source.env_callback.save_on_step_callback import SaveOnStepCallback


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

    def key_info(self) -> str:
        return (
            "[조작법] A: 왼쪽, D: 오른쪽, ESC: 종료\n"
            "키보드로 조작하거나, ESC를 눌러 종료할 수 있습니다."
        )

    def _self_play(self):
        self.reset()
        running = True
        action = None  # None이면 랜덤, 0: left, 1: right
        clock = pygame.time.Clock()
        pygame.display.set_caption(
            "CartPole-v1 Controller (A: Left, D: Right, ESC: Quit)"
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
        save_path = os.path.join(self.save_dir, "ppo_cartpole.zip")

        model = PPO("MlpPolicy", self.env, verbose=1)
        total_timesteps = 1000000

        callback = SaveOnStepCallback(
            save_freq=10000,
            logging_freq=1000,
            save_path=self.save_dir,
            name_prefix="ppo_cartpole",
            log_dir=log_dir,
            verbose=1,
        )
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
        )
        model.save(save_path)
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
