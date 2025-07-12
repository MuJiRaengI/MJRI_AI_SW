import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time
import json
import torch
import pygame
import cv2
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from source.ai.rl.model.find_avoid_observer_model import (
    AvoidStoppedObserverExtractor,
    DoubleResnet18,
    UNet,
)
from source.envs.env_starcraft import EnvStarcraft
import keyboard


class StarcraftAvoidObserver(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "StarcraftAvoidObserver-v0"
        self.fps = 60
        self.running = False
        self.device = "cuda:0"
        keyboard.add_hotkey("ctrl+p", self.stop_running)
        keyboard.add_hotkey("f4", self.stop_running)

    def stop_running(self):
        self.running = False

    def key_info(self) -> str:
        return "[조작법] 기본 스타 조작법\n" "ESC: 종료\n"

    def _self_play(self, *args, **kwargs):
        # load direction ai
        direction_ai = DoubleResnet18(3, 4)
        direction_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\direct_ai.pth"
        direction_ai.load_state_dict(torch.load(direction_ai_path))
        direction_ai.to(self.device)
        direction_ai.eval()
        print(f"load path classification model ({direction_ai_path})")

        detect_ai = UNet(3, 4)
        detect_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\detect_ai.pth"
        detect_ai.load_state_dict(torch.load(detect_ai_path))
        detect_ai.to(self.device)
        detect_ai.eval()
        print(f"load object detection model ({detect_ai_path})")

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

            with torch.no_grad():
                tmr = time.time()
                now_scene = env.capture()
                game_scene, minimap, _ = env.split_map(now_scene)
                direct_game_scene, direct_minimap = env.transform_direction(
                    game_scene, minimap
                )
                direct_game_scene = direct_game_scene.to(self.device)
                direct_minimap = direct_minimap.to(self.device)
                direction = direction_ai(direct_game_scene, direct_minimap)
                direction = direction.argmax().item()  # 0:down, 1:left, 2:right, 3:up
                print(f"Predicted direction: {direction}")

                x_crop, y_crop, w_crop, h_crop = 400, 100, 512, 512
                game_scene_crop = game_scene[
                    y_crop : y_crop + h_crop, x_crop : x_crop + w_crop
                ]
                detect_scene = env.transform_detection(game_scene_crop)
                detect_scene = detect_scene.to(self.device)
                detect = detect_ai(detect_scene)

                # 1 : observer
                # 2 : my
                # 3 : path
                threshold = 0.99
                detect = detect > threshold
                detect = detect * 255
                detect = detect.detach().cpu().numpy().astype(np.uint8)

                bg = detect[0, 0]
                observer = detect[0, 1]
                # char = detect[0, 2]
                # path = detect[0, 3]

                # postprocessing
                kernel = np.ones((5, 5), np.uint8)
                # observer = cv2.morphologyEx(observer, cv2.MORPH_OPEN, kernel)
                observer = cv2.morphologyEx(
                    observer, cv2.MORPH_CLOSE, kernel, iterations=2
                )
                bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel, iterations=2)

                print("time : {:.3f}s".format(time.time() - tmr))
                print()

            clock.tick(self.fps)

        env.stop_focusing()
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

        env.start_focusing()
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
                env.stop_focusing()
                obs = env.reset()

            clock.tick(self.fps)

        env.stop_focusing()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self, *args, **kwargs):
        # load direction ai
        direction_ai = DoubleResnet18(3, 4)
        direction_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\direct_ai.pth"
        direction_ai.load_state_dict(torch.load(direction_ai_path))
        direction_ai.to(self.device)
        direction_ai.eval()
        print(f"load path classification model ({direction_ai_path})")

        detect_ai = UNet(3, 4)
        detect_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\detect_ai.pth"
        detect_ai.load_state_dict(torch.load(detect_ai_path))
        detect_ai.to(self.device)
        detect_ai.eval()
        print(f"load object detection model ({detect_ai_path})")

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

            with torch.no_grad():
                now_scene = env.capture()
                game_scene, minimap, _ = env.split_map(now_scene)
                direct_game_scene, direct_minimap = env.transform_direction(
                    game_scene, minimap
                )
                direct_game_scene = direct_game_scene.to(self.device)
                direct_minimap = direct_minimap.to(self.device)
                direction = direction_ai(direct_game_scene, direct_minimap)
                direction = direction.argmax().item()  # 0:down, 1:left, 2:right, 3:up
                print(f"Predicted direction: {direction}")

                x_crop, y_crop, w_crop, h_crop = 400, 100, 512, 512
                game_scene_crop = game_scene[
                    y_crop : y_crop + h_crop, x_crop : x_crop + w_crop
                ]
                detect_scene = env.transform_detection(game_scene_crop)
                detect_scene = detect_scene.to(self.device)
                detect = detect_ai(detect_scene)

                # 1 : observer
                # 2 : my
                # 3 : path
                threshold = 0.99
                detect = detect > threshold
                detect = detect * 255
                detect = detect.detach().cpu().numpy().astype(np.uint8)

                bg = detect[0, 0]
                observer = detect[0, 1]
                # char = detect[0, 2]
                # path = detect[0, 3]

                # postprocessing
                kernel = np.ones((5, 5), np.uint8)
                # observer = cv2.morphologyEx(observer, cv2.MORPH_OPEN, kernel)
                observer = cv2.morphologyEx(
                    observer, cv2.MORPH_CLOSE, kernel, iterations=2
                )
                bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel, iterations=2)

            clock.tick(self.fps)

        env.stop_focusing()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

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
