import gym
from gym import spaces
import numpy as np
import pygame
import torch
from PIL import Image
from collections import deque
import cv2
import torch.nn.functional as F
import os
import time

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

        self.mouse.drag(
            self.x + 10,
            self.y + 10,
            self.x + 1100,
            self.y + 600,
        )
        self.keyboard.set_numbering(self.focus_num)

    # def start_focusing(self):

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


if __name__ == "__main__":
    env = EnvStarcraft()
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.close()
