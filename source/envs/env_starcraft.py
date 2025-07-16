# import gym
import gymnasium as gym

# from gym import spaces
import gymnasium.spaces as spaces
import numpy as np
import pygame
import torch
from PIL import Image
from collections import deque
import cv2
import torch.nn.functional as F
import os
import time
import threading

from source.utils.mjri_screen import MJRIScreen
from source.utils.mjri_keyboard import MJRIKeyboard
from source.utils.mjri_mouse import MJRIMouse


class EnvStarcraft(gym.Env):
    def __init__(self, env_id: str):
        super().__init__()
        self.env_id = env_id
        self.max_buffer_size = None
        if self.env_id == "StarcraftAvoidObserver-v0":
            self.max_buffer_size = 4

        self.focus_num = 1
        self.focus_thread = None
        self.focus_flag = False

        self.keyboard = MJRIKeyboard()
        self.mouse = MJRIMouse()
        self.screen = MJRIScreen(self.max_buffer_size)

        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0
        self.char_h_margin = -90
        self.ready = False

    def reset(self, *args, **kwargs):
        if not self.ready:
            raise ValueError(
                "Screen position not set. Use set_screen_pos() before reset()."
            )
        self.mouse.leftClick(self.x + 30, self.y + 30, delay=0.1)

        self.mouse.drag(
            self.x + 30,
            self.y + 30,
            self.x + 1100,
            self.y + 600,
        )
        self.keyboard.set_numbering(self.focus_num)

    def start_focusing(self):
        self.focus_thread = threading.Thread(
            target=self.focus_numbering_auto, args=(self.focus_num,)
        )
        self.focus_thread.start()

    def focus_numbering_auto(self, number):
        self.focus_flag = True
        while self.focus_flag:
            self.keyboard.focus_numbering(str(number))

    def stop_focusing(self):
        self.focus_flag = False

    def set_screen_pos(self, x: int, y: int, w: int, h: int):
        """화면의 특정 영역을 설정합니다."""
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.screen.set_screen_pos(x, y, w, h)
        self.ready = True

    def step(self, action: int):
        self.unit_move_with_angle(action * 45, 150)
        return None, None, None, None

    def get_char_pos(self):
        """캐릭터의 현재 위치를 반환합니다."""
        char_x = self.x + self.w // 2
        char_y = self.y + self.h // 2 + self.char_h_margin
        return char_x, char_y

    def capture(self):
        screenshot = self.screen.capture()
        return screenshot

    def unit_move_with_angle(self, best_angle, move_range=1):
        """캐릭터를 특정 각도로 이동시킵니다."""
        char_x, char_y = self.get_char_pos()

        rad = np.deg2rad(best_angle)
        dx = int(np.cos(rad) * move_range)
        dy = int(np.sin(rad) * move_range)

        target_x = char_x + dx
        target_y = char_y + dy

        self.mouse.rightClick(target_x, target_y, delay=0.1)
        return

    def render(self, mode="human"):
        pass

    def crop_game_screen(self, scene):
        return scene[35:650, 3:-3]

    def crop_minimap(self, scene):
        return scene[720:-3, 3:280]

    def crop_now_unit_info(self, scene):
        return scene[780:, 280:810]

    def split_map(self, screen):
        game_screen = self.crop_game_screen(screen)
        minimap = self.crop_minimap(screen)
        unit_info = self.crop_now_unit_info(screen)
        return game_screen, minimap, unit_info

    def transform_direction(self, game_scene, minimap, downsample=3):
        game_scene = torch.from_numpy(game_scene).permute(2, 0, 1).float()
        game_scene = game_scene / 255.0
        minimap = torch.from_numpy(minimap).permute(2, 0, 1).float()
        minimap = minimap / 255.0

        if downsample > 1:
            height, width = game_scene.shape[1:]
            new_size = (height // downsample, width // downsample)
            game_scene = torch.nn.functional.interpolate(
                game_scene.unsqueeze(0),
                size=new_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        return game_scene.unsqueeze(0), minimap.unsqueeze(0)

    def transform_detection(self, scene):
        scene = torch.from_numpy(scene).permute(2, 0, 1).float()
        scene = scene / 255.0
        return scene.unsqueeze(0)


if __name__ == "__main__":
    env = EnvStarcraft()
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.close()
