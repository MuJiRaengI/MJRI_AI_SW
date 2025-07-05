import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time
import json
import torch
import pygame
import numpy as np
from collections import OrderedDict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from source.ai.rl.model.find_avoid_observer_model import AvoidStoppedObserverExtractor
from source.envs.env_starcraft import EnvStarcraft

import keyboard


class StarcraftAvoidObserver(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "StarcraftAvoidObserver-v0"
        self.fps = 60
        self.running = False
        keyboard.add_hotkey("ctrl+p", self.stop_running)
        keyboard.add_hotkey("f4", self.stop_running)

    def stop_running(self):
        self.running = False

    def key_info(self) -> str:
        return "[조작법] 기본 스타 조작법\n" "ESC: 종료\n"

    def _self_play(self, *args, **kwargs):
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self, *args, **kwargs):
        env = EnvStarcraft(self.env_id)
        screen_pos = kwargs.get("screen_pos", None)
        if screen_pos is not None:
            x, y, w, h = screen_pos
            env.set_screen_pos(x, y, w, h)
        else:
            raise ValueError("screen_pos must be provided for random play.")

        obs = env.reset()
        pygame.init()
        env.render()
        pygame.display.set_caption("Starcraft Avoid Observer Random Play (ESC: Quit)")
        clock = pygame.time.Clock()

        self.running = True
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False

            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 랜덤 액션
            # action = env.action_space.sample()
            action = np.random.randint(0, 8)  # 0~7 사이의 랜덤 액션
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                obs = env.reset()

            clock.tick(self.fps)

        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self, *args, **kwargs):
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

    def _test(self, *args, **kwargs):
        if self.training_queue is not None:
            self.training_queue.put(("done", None))


if __name__ == "__main__":
    find_avoid_observer = FindAvoidObserver()
    find_avoid_observer.save_dir = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Find_op_map_path"
    )
    find_avoid_observer.log_dir = "logs"
    find_avoid_observer._train()
