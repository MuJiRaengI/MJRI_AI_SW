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
from source.ai.rl.BBF_agent.BBF import BBF
from source.ai.rl.model.avoid_observer_bbf import BBF_Model
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from source.ai.rl.model.find_avoid_observer_model import (
    AvoidObserverExtractor,
    DoubleResnet18,
    UNet,
)
from source.envs.env_starcraft import EnvStarcraft
import keyboard
from collections import deque


class StarcraftAvoidObserver(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "StarcraftAvoidObserver-v0"
        self.agent = None
        self.total_timesteps = 100000
        self.save_freq = 10000
        self.logging_freq = 1000
        self.feature_dim = 256

        self.max_step_length = 2000
        self.running = False
        self.device = "cuda:0"
        keyboard.add_hotkey("ctrl+p", self.stop_running)
        keyboard.add_hotkey("f4", self.stop_running)

    def stop_running(self):
        self.running = False
        print("Stopping the game...")

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

        screen_pos = kwargs.get("screen_pos", None)
        env = EnvStarcraft(self.env_id, screen_pos=screen_pos)

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
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        env = EnvStarcraft(self.env_id, self.max_step_length)
        screen_pos = kwargs.get("screen_pos", None)
        if screen_pos is not None:
            x, y, w, h = screen_pos
            env.set_screen_pos(x, y, w, h)
        else:
            raise ValueError("screen_pos must be provided for random play.")

        n_actions = env.action_space.n

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        model = BBF_Model(
            n_actions,  # env action space size
            hiddens=2048,  # representation dim
            scale_width=4,  # cnn channel scale
            num_buckets=51,  # buckets in distributional RL
            Vmin=-2,  # min value in distributional RL
            Vmax=30,  # max value in distributional RL
            resize=(80, 80),  # input resize
        ).cuda()

        agent = BBF(
            model,
            env,
            learning_rate=1e-4,
            batch_size=32,
            ema_decay=0.995,  # target model ema decay
            initial_gamma=0.97,  # starting gamma
            final_gamma=0.997,  # final gamma
            initial_n=10,  # starting n-step
            final_n=3,  # final n-step
            num_buckets=51,  # buckets in distributional RL
            reset_freq=40000,  # reset schedule in grad step
            replay_ratio=2,  # update number in one step
            weight_decay=0.1,  # weight decay in optimizer,
            epsilon=0,
            gym_env=True,
            stackFrame=False,
        )

        model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Starcraft_avoid_observer\bbf_avoid_observer_190000_steps.pth"
        if os.path.exists(model_path):
            print(f"기존 모델을 불러옵니다: {model_path}")
            agent.load(model_path)

        env.start_focusing()
        agent.learn(
            total_timesteps=self.total_timesteps,
            save_freq=self.save_freq,
            save_path=self.save_dir,
            name_prefix="bbf_avoid_observer",  # save file name prefix
        )
        env.stop_focusing()

        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _test(self, *args, **kwargs):
        last_model_path = None
        model = None
        count = 0

        screen_pos = kwargs.get("screen_pos", None)
        env = EnvStarcraft(self.env_id, screen_pos=screen_pos)

        n_actions = env.action_space.n
        obs = env.reset()

        model = BBF_Model(n_actions, resize=(80, 80)).cuda()  # input resize
        state = model.preprocess(obs).unsqueeze(0)

        # 초기 렌더링을 위해 단일 환경에 접근
        env.render()

        self.running = True
        env.start_focusing()
        while self.running:
            count += 1
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 모델 파일 탐색 및 필요시 reload
            model_path = os.path.join(self.save_dir, "bbf_avoid_observer.pth")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"bbf_avoid_observer_(\d+)_steps\.pth", fname)
                    if match:
                        steps = int(match.group(1))
                        if steps > max_steps:
                            max_steps = steps
                            max_steps_path = os.path.join(self.save_dir, fname)
                if max_steps_path:
                    model_path = max_steps_path

            if model_path != last_model_path and os.path.exists(model_path):
                time.sleep(0.5)  # 잠시 대기 후 모델 로드
                print(f"모델 업데이트: {model_path}")
                model.load(model_path)
                last_model_path = model_path

            if last_model_path is None:
                print("테스트 가능한 모델 파일이 없습니다. (기본 BBF로 테스트)")
                last_model_path = ""

            # 모델이 제대로 로드되었을 때만 예측 수행
            if model is not None:
                # 모델 예측
                r_state1 = state[0, :3]
                r_state2 = state[0, 3:6]
                r_state3 = state[0, 6:9]
                r_state4 = state[0, 9:12]
                r_state5 = state[0, 12:15]

                # if count > 30:
                #     print("")
                action = model.predict(state.unsqueeze(0))
                obs, reward, done, info = env.step(action.cpu().item())
                # print(f"Reward: {round(reward, 2)}", end="\r", flush=True)
                state = model.preprocess(obs).unsqueeze(0)
                env.render()

                if done:
                    obs = env.reset()
                    state = model.preprocess(obs).unsqueeze(0)
            else:
                # 모델이 없으면 대기 (화면은 계속 렌더링)
                env.render()

                # 모델 로딩 대기 메시지 표시를 위해 잠시 대기
                time.sleep(0.5)

        env.stop_focusing()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    find_avoid_observer = FindAvoidObserver()
    find_avoid_observer.save_dir = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Find_op_map_path"
    )
    find_avoid_observer.log_dir = "logs"
    find_avoid_observer._train()
