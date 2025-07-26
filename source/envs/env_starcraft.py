# import gym
import gymnasium as gym

# from gym import spaces
import gymnasium.spaces as spaces
import numpy as np
import torch
from PIL import Image
from collections import deque
import cv2
import torch.nn.functional as F
import os
import time
import threading
from datetime import datetime

from source.utils.mjri_screen import MJRIScreen
from source.utils.mjri_keyboard import MJRIKeyboard
from source.utils.mjri_mouse import MJRIMouse
from source.ai.rl.model.find_avoid_observer_model import DoubleResnet18, UNet


class EnvStarcraft(gym.Env):
    def __init__(
        self,
        env_id: str,
        max_steps: int = 1000,
        scale_factor: float = 0.25,
        frame_stack: int = 4,
        fps=30,
        screen_pos: tuple = None,
    ):
        super().__init__()
        self.env_id = env_id
        self.max_steps = max_steps
        self.scale_factor = scale_factor
        self.frame_stack = frame_stack
        self.screen_pos = screen_pos
        self.fps = fps
        self.action_space = spaces.Discrete(8)
        self.max_buffer_size = None
        if self.env_id == "StarcraftAvoidObserver-v0":
            self.max_buffer_size = 4

        self.ready = False

        self.focus_num = 1
        self.focus_thread = None
        self.focus_flag = False

        self.keyboard = MJRIKeyboard()
        self.mouse = MJRIMouse()
        self.screen = MJRIScreen(self.max_buffer_size, thread_run=True, fps=fps)
        self.screen.set_buffer(self.max_buffer_size)

        self.x = 0
        self.y = 0
        self.w = 0
        self.h = 0

        if screen_pos is not None:
            x, y, w, h = screen_pos
            self.set_screen_pos(x, y, w, h)

        self.x_crop, self.y_crop, self.w_crop, self.h_crop = 400, 100, 512, 512

        self.char_h_margin = -90

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

        for i in range(self.max_buffer_size):
            self.screen.capture()
            time.sleep(0.1)  # wait for the screen to capture

        self.get_stacked_state()

    def __del__(self):
        if self.focus_thread is not None:
            self.focus_flag = False
            self.focus_thread.join()
        self.screen.stop()

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
        self.keyboard.focus_numbering(str(self.focus_num))

        self.steps = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None

        return self._get_obs()

    def get_state_tensor(self, now_scene=None):
        with torch.no_grad():
            if now_scene is None:
                now_scene = self.capture()
            game_scene, minimap, _ = self.split_map(now_scene)
            direction = self.get_direction_one_hot(game_scene, minimap)
            direction = direction.argmax().item()  # 0:down, 1:left, 2:right, 3:up
            game_scene_crop = game_scene[
                self.y_crop : self.y_crop + self.h_crop,
                self.x_crop : self.x_crop + self.w_crop,
            ]
            detect_scene = self.transform_detection(game_scene_crop)
            detect_scene = detect_scene.to(self.device)
            detect = self.detect_ai(detect_scene)

            # postprocessing
            threshold = 0.99
            detect = detect > threshold
            detect = detect * 255
            detect = detect.detach().cpu().numpy().astype(np.uint8)

            # real
            # bg = detect[0, 0]
            # observer = detect[0, 1]
            # char = detect[0, 2]

            # kernel = np.ones((5, 5), np.uint8)
            # observer = cv2.morphologyEx(observer, cv2.MORPH_OPEN, kernel)
            # observer = cv2.morphologyEx(observer, cv2.MORPH_CLOSE, kernel, iterations=2)
            # bg = cv2.morphologyEx(bg, cv2.MORPH_OPEN, kernel, iterations=2)

            # frame = np.zeros((3, bg.shape[0], bg.shape[1]), dtype=np.uint8)

            # frame[0] = bg
            # frame[1] = observer
            # frame[2] = char

            # virtual
            radius = 15
            bg = detect[0, 0]

            observer = np.zeros_like(detect[0, 1])
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                (detect[0, 1] == 255).astype(np.uint8)
            )
            for i in range(1, num_labels):
                center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
                cv2.circle(observer, (center_x, center_y), radius, 255, -1)

            char = np.zeros_like(detect[0, 2])
            cv2.circle(char, (self.w_crop // 2, self.h_crop // 2), radius, 255, -1)
            # ys, xs = np.where(detect[0, 2] == 255)
            # if len(xs) > 0 and len(ys) > 0:
            #     center_x = int(xs.mean())
            #     center_y = int(ys.mean())
            #     cv2.circle(char, (center_x, center_y), radius, 255, -1)

            frame = np.zeros((3, bg.shape[0], bg.shape[1]), dtype=np.uint8)
            frame[0] = bg
            frame[1] = observer
            frame[2] = char

        frame = torch.from_numpy(frame)
        return frame

    def get_stacked_state(self):
        stacked_frame = self.screen.screenshot_buffer
        # stacked_frame = np.concat(stacked_frame, axis=-1)
        stacked_frame = np.stack(stacked_frame, axis=3)
        stacked_game_scene, _, _ = self.split_map(stacked_frame)
        # crop
        stacked_game_scene = stacked_game_scene[
            self.y_crop : self.y_crop + self.h_crop,
            self.x_crop : self.x_crop + self.w_crop,
        ]
        stacked_game_scene = torch.from_numpy(stacked_game_scene).permute(3, 2, 0, 1)
        # stacked_frame_resize = F.interpolate(
        #     stacked_game_scene,
        #     scale_factor=(self.scale_factor, self.scale_factor),
        #     mode="nearest",
        # ).squeeze()
        return stacked_game_scene

    def _get_image_obs(self):
        state = self.get_stacked_state()

        with torch.no_grad():
            detect_scene = self.transform_detection(state)
            detect_scene = detect_scene.to(self.device)
            detect = self.detect_ai(detect_scene)

            threshold = 0.99
            detect = detect > threshold
            detect = detect * 255
            detect = detect.detach().cpu().numpy().astype(np.uint8)

            radius = 17
            frames = np.zeros(
                (3 * detect.shape[0], detect.shape[2], detect.shape[3]), dtype=np.uint8
            )
            char = np.zeros_like(detect[0, 2])
            cv2.circle(char, (self.w_crop // 2, self.h_crop // 2), radius, 255, -1)

            for idx in range(detect.shape[0]):
                frame = np.zeros((3, detect.shape[2], detect.shape[3]), dtype=np.uint8)

                bg = detect[idx, 0]
                observer = np.zeros_like(detect[idx, 1])
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                    (detect[idx, 1] == 255).astype(np.uint8)
                )
                for i in range(1, num_labels):
                    center_x, center_y = int(centroids[i][0]), int(centroids[i][1])
                    cv2.circle(observer, (center_x, center_y), radius, 255, -1)

                frame[0] = bg
                frame[1] = observer
                frame[2] = char

                frames[idx * 3 : (idx + 1) * 3] = frame

        frames = torch.from_numpy(frames)
        frames = F.interpolate(
            frames.unsqueeze(0),
            scale_factor=(self.scale_factor, self.scale_factor),
            mode="nearest",
        ).squeeze()
        return frames

    def _get_vector_obs(self):
        return np.array(self.get_direction_one_hot())

    def get_direction_one_hot(self, game_scene=None, minimap=None):
        if game_scene is None or minimap is None:
            now_scene = self.screen.screenshot_buffer[-1]
            game_scene, minimap, _ = self.split_map(now_scene)
        direct_game_scene, direct_minimap = self.transform_direction(
            game_scene, minimap
        )

        direct_game_scene = direct_game_scene.to(self.device)
        direct_minimap = direct_minimap.to(self.device)

        direction = self.direction_ai(direct_game_scene, direct_minimap)
        direction = direction.argmax().item()  # 0:down, 1:left, 2:right, 3:up

        vector = np.zeros((4,), dtype=np.float32)
        if direction == 0:  # down
            vector[1] = 1.0
        elif direction == 1:  # left
            vector[2] = 1.0
        elif direction == 2:  # right
            vector[0] = 1.0
        elif direction == 3:  # up
            vector[3] = 1.0

        return vector

    def _get_obs(self):
        with torch.no_grad():
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

    def calc_reward(self, action: int, now_scene: np.ndarray):
        done = False
        game_scene, minimap, unit_info = self.split_map(now_scene)
        unit_die = self.unit_die(unit_info)
        unit_reset = self.unit_reset(game_scene)

        # unit die
        if unit_die or unit_reset:
            time.sleep(2)
            # check goal
            if self.check_goal(game_scene):
                now_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                os.mkdir(f"goal_{now_time}")
                return 1.0, True

            self.keyboard.press_and_release("s", delay=0.1)
            return -0.5, True

        direction = np.array(self.get_direction_one_hot()).argmax()
        degree = [np.pi / 4 * i for i in range(8)]
        reward = np.cos(degree[direction * 2] - degree[action]) * 0.1

        if self.steps >= self.max_steps:
            done = True
        return reward, done

    def step(self, action: int):
        self.last_action = action
        if not hasattr(self, "total_reward"):
            self.total_reward = 0.0

        self.unit_move_with_angle(action * 45, 300)
        # time.sleep(0.1)

        # now_scene = self.capture()
        now_scene = self.screen.screenshot_buffer[-1]

        self.reward, done = self.calc_reward(action, now_scene)
        self.total_reward += self.reward
        self.steps += 1

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

        self.mouse.rightClick(target_x, target_y, delay=0.005)
        return

    def render(self, mode="human"):
        pass

    def check_goal(self, frame):
        crop_frame = frame[
            self.y_crop : self.y_crop + self.h_crop,
            self.x_crop : self.x_crop + self.w_crop,
        ]

        scout_template_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\source\envs\starcraft\template\scout"
        templates = os.listdir(scout_template_path)

        for template in templates:
            template_path = os.path.join(scout_template_path, template)
            template_img = cv2.imread(template_path)
            res = cv2.matchTemplate(crop_frame, template_img, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)

            if len(loc[0]) > 0:
                return True

        return False

    def unit_die(self, unit_info):
        g = unit_info[..., 1]
        green_count = (g > 200).sum()
        return green_count < 500

    def unit_reset(self, frame):
        y_center = self.y_crop + self.h_crop // 2
        x_center = self.x_crop + self.w_crop // 2
        margin = 256
        crop_frame = frame[
            y_center - margin : y_center + margin,
            x_center - margin : x_center + margin,
        ]
        b, g, r = cv2.split(crop_frame)
        # mask = (r >= 100) & (b <= 20) & (g <= 20)
        # r > b + g + 20
        mask = r.astype(np.float32) > np.clip(
            b.astype(np.float32) + g.astype(np.float32) + 30, 0, 255
        )
        return mask.sum() < 100

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
        if isinstance(scene, np.ndarray):
            scene = torch.from_numpy(scene).permute(2, 0, 1)
        scene = scene.float() / 255.0
        if scene.dim() == 3:
            scene = scene.unsqueeze(0)
        return scene


if __name__ == "__main__":
    env = EnvStarcraft()
    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # 랜덤 액션
        obs, reward, done, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
    env.close()
