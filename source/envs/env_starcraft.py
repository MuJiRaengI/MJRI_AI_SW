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
from source.ai.rl.model.find_avoid_observer_model import DoubleResnet18, UNet


class EnvStarcraft(gym.Env):
    def __init__(
        self,
        env_id: str,
        scale_factor: float = 0.25,
        frame_stack: int = 4,
    ):
        super().__init__()
        self.env_id = env_id
        self.scale_factor = scale_factor
        self.frame_stack = frame_stack
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

        self.steps = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None

        # load direction ai
        self.device = "cuda:0"
        self.direction_ai = DoubleResnet18(3, 4)
        self.direction_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\direct_ai.pth"
        self.direction_ai.load_state_dict(torch.load(self.direction_ai_path))
        self.direction_ai.to(self.device)
        self.direction_ai.eval()
        print(f"load path classification model ({self.direction_ai_path})")

        self.detect_ai = UNet(3, 4)
        self.detect_ai_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\detect_ai.pth"
        self.detect_ai.load_state_dict(torch.load(self.detect_ai_path))
        self.detect_ai.to(self.device)
        self.detect_ai.eval()
        print(f"load object detection model ({self.detect_ai_path})")

        self.state_buffer = deque(maxlen=self.frame_stack)
        self.x_crop, self.y_crop, self.w_crop, self.h_crop = 400, 100, 512, 512

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

        self.steps = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None

        state = self.get_state_tensor()
        self.state_buffer.extend([state] * 3)

        return self._get_obs()

    def get_state_tensor(self, scene=None):
        if scene is None:
            scene = self.capture()

        # crop
        scene = np.swapaxes(scene, 2, 1)
        scene = np.swapaxes(scene, 1, 0)
        scene = scene[
            :,
            self.y_crop : self.y_crop + self.h_crop,
            self.x_crop : self.x_crop + self.w_crop,
        ]

        return scene

    def get_stacked_state(self):
        return F.interpolate(
            torch.concat(torch.tensor(list(self.state_buffer)), dim=0).unsqueeze(0),
            scale_factor=(self.scale_factor, self.scale_factor),
            mode="nearest",
        ).squeeze()

    def _get_image_obs(self):
        state = self.get_stacked_state()
        return np.array(state)

    def _get_vector_obs(self, game_scene):
        return np.array(self.get_direction_one_hot(game_scene))

    def get_direction_one_hot(self, game_scene, minimap):
        direct_game_scene, direct_minimap = self.transform_direction(
            game_scene, minimap
        )

        direct_game_scene = direct_game_scene.to(self.device)
        direct_minimap = direct_minimap.to(self.device)

        direction = self.direction_ai(direct_game_scene, direct_minimap)
        return direction

    def _get_obs(self):
        with torch.no_grad():
            now_scene = self.get_state_tensor()
            game_scene, minimap, _ = self.split_map(now_scene)
            direction = self.get_direction_one_hot(game_scene, minimap)
            direction = direction.argmax().item()  # 0:down, 1:left, 2:right, 3:up

            x_crop, y_crop, w_crop, h_crop = 400, 100, 512, 512
            game_scene_crop = game_scene[
                y_crop : y_crop + h_crop, x_crop : x_crop + w_crop
            ]
            detect_scene = self.transform_detection(game_scene_crop)
            detect_scene = detect_scene.to(self.device)
            detect = self.detect_ai(detect_scene)

            # postprocessing
            threshold = 0.99
            detect = detect > threshold
            detect = detect * 255
            detect = detect.detach().cpu().numpy().astype(np.uint8)

            bg = detect[0, 0]
            observer = detect[0, 1]
            char = detect[0, 2]

            kernel = np.ones((5, 5), np.uint8)
            observer = cv2.morphologyEx(observer, cv2.MORPH_OPEN, kernel)
            observer = cv2.morphologyEx(observer, cv2.MORPH_CLOSE, kernel, iterations=2)
            bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel, iterations=2)

            frame = np.zeros((bg.shape[0], bg.shape[1], 3), dtype=np.uint8)

            frame[:, :, 2] = bg
            frame[:, :, 1] = observer
            frame[:, :, 0] = char

        # obs 생성
        image_obs = self._get_image_obs()
        vector_obs = self._get_vector_obs()
        obs = {
            "image": np.array(image_obs, dtype=np.uint8),
            "vector": np.array(vector_obs, dtype=np.float32),
        }
        return obs

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

    def calc_reward(self, action: int, state):

        # goal
        # return 1.0

        # die
        # return -0.5

        direction = np.array(self.get_direction_one_hot()).argmax()
        degree = [np.pi / 4 * i for i in range(8)]
        reward = np.cos(degree[direction * 2] - degree[action]) * 0.1

        return reward

    def step(self, action: int):
        self.unit_move_with_angle(action * 45, 150)

        state = self.capture()

        self.reward = self.calc_reward(action, state)
        self.total_reward += self.reward
        self.steps += 1

        # 종료조건

        self.state_buffer.append(state)

        return self._get_obs(), self.reward, done, {}

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
