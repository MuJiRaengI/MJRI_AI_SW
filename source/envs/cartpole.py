import os
import sys

sys.path.append(os.path.abspath("."))

import random
import re
import time
import gymnasium as gym
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
        self.env_id = "CartPole-v1"
        self.total_timesteps = 1000000
        self.n_envs = 8

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽\n"

    def _self_play(self):
        # CartPole-v1 환경 생성 및 수동 플레이 (A, D키로 조작)
        env = gym.make(self.env_id, render_mode="human")
        obs, info = env.reset()
        clock = pygame.time.Clock()
        pygame.display.set_caption("CartPole-v1 Manual Play (A: Left, D: Right)")
        while True:
            # 윈도우 X(닫기) 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return

            # render_queue로부터 stop 신호를 받으면 중단
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 키보드 입력 처리
            action = None
            if keyboard.is_pressed("a"):
                action = 0  # 왼쪽
            elif keyboard.is_pressed("d"):
                action = 1  # 오른쪽
            else:
                action = env.action_space.sample()  # 아무 키도 안 누르면 랜덤
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
            clock.tick(getattr(self, "fps", 30))
        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self):
        # CartPole-v1 환경 생성 및 랜덤 플레이
        env = gym.make(self.env_id, render_mode="human")
        obs, info = env.reset()
        clock = pygame.time.Clock()
        pygame.display.set_caption("CartPole-v1 Random Play")
        while True:
            # 윈도우 X(닫기) 이벤트 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return

            # render_queue로부터 stop 신호를 받으면 중단
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            action = env.action_space.sample()  # 랜덤 액션
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
            clock.tick(getattr(self, "fps", 30))
        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self):
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        env = make_vec_env(self.env_id, n_envs=self.n_envs)

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        callback = SaveOnStepCallback(
            save_freq=10000,
            logging_freq=10000,
            save_dir=self.save_dir,
            name_prefix="ppo_cartpole",
            log_dir=log_dir,
            progress_queue=self.training_queue,
            verbose=1,
        )

        # 모델 생성 및 학습
        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

        # 모델 저장
        save_path = os.path.join(self.save_dir, "ppo_cartpole.zip")
        tmp_path = save_path.replace("zip", "tmp")
        model.save(tmp_path)
        os.replace(tmp_path, save_path)
        print(f"모델 저장 완료: {save_path}")

    def _test(self):
        last_model_path = None
        model = None
        env = gym.make(self.env_id, render_mode="human")
        obs, info = env.reset()
        clock = pygame.time.Clock()
        pygame.display.set_caption("CartPole-v1 Test")
        while True:
            # 모델 파일 탐색 및 필요시 reload
            model_path = os.path.join(self.save_dir, "ppo_cartpole.zip")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"ppo_cartpole_(\d+)_steps.zip", fname)
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
                model = PPO.load(model_path)
                last_model_path = model_path
            elif model is None:
                print("테스트 가능한 모델 파일이 없습니다. (기본 PPO로 테스트)")
                model = PPO("MlpPolicy", gym.make(self.env_id), device="cpu")
                last_model_path = None

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            if terminated or truncated:
                obs, info = env.reset()
                # 에피소드가 끝나도 env는 유지, 모델만 reload
                continue
            clock.tick(getattr(self, "fps", 30))
        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))
