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
from .env_avoid_observber import EnvAvoidObserver


class AvoidStoppedObserver(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "FindAvoidObserver-v1"
        self.total_timesteps = 10000000
        self.save_freq = 1000
        self.logging_freq = 1000
        self.feature_dim = 256
        self.n_envs = 4
        self.scale = 1
        self.deterministic = False

        self.max_step_length = 1000

    def key_info(self) -> str:
        return "[조작법] D(→), C(↘), S(↓), Z(↙), A(←), Q(↖), W(↑), E(↗)\n" "ESC: 종료\n"

    def _self_play(self):
        env = EnvAvoidObserver(
            max_steps=self.max_step_length,
            num_observers=50,
            random_start=False,
            move_observer=True,
        )
        obs = env.reset()

        pygame.init()
        env.render()  # 초기 렌더링으로 화면 크기 설정
        pygame.display.set_caption(
            "FindAvoidObserver Manual Play (D/C/S/Z/A/Q/W/E: Move, ESC: Quit)"
        )
        clock = pygame.time.Clock()

        running = True
        action = None

        while running:
            # 윈도우 종료 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # render_queue로부터 stop 신호를 받으면 중단
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 키보드 입력 처리 (pygame 방식)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_d]:
                action = 0  # RIGHT
            elif keys[pygame.K_c]:
                action = 1  # DOWN_RIGHT
            elif keys[pygame.K_s]:
                action = 2  # DOWN
            elif keys[pygame.K_z]:
                action = 3  # DOWN_LEFT
            elif keys[pygame.K_a]:
                action = 4  # LEFT
            elif keys[pygame.K_q]:
                action = 5  # UP_LEFT
            elif keys[pygame.K_w]:
                action = 6  # UP
            elif keys[pygame.K_e]:
                action = 7  # UP_RIGHT

            # 환경 업데이트
            obs, reward, done, info = env.step(action)
            env.render()

            # 에피소드 끝났으면 리셋
            if done:
                obs = env.reset()

            clock.tick(self.fps)

        env.stop_recording()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self):
        env = EnvAvoidObserver(
            max_steps=self.max_step_length,
            num_observers=50,
            random_start=False,
            move_observer=True,
        )
        obs = env.reset()

        pygame.init()
        env.render()
        pygame.display.set_caption("FindAvoidObserver Random Play (ESC: Quit)")
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 랜덤 액션
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            env.render()

            if done:
                obs = env.reset()

            clock.tick(self.fps)

        env.stop_recording()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self):
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 커스텀 환경 래핑
        def make_env():
            return EnvAvoidObserver(
                max_steps=self.max_step_length,
                num_observers=50,
                random_start=True,
                move_observer=True,
            )

        # 여러 환경으로 병렬 학습
        env = DummyVecEnv([make_env for _ in range(self.n_envs)])

        # VecMonitor 추가로 episode 통계 수집
        env = VecMonitor(env)

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        callback = SaveOnStepCallback(
            save_freq=self.save_freq,
            logging_freq=self.logging_freq,
            save_dir=self.save_dir,
            name_prefix="ppo_find_avoid_observer",
            log_dir=log_dir,
            progress_queue=self.training_queue,
            verbose=1,
        )

        # 모델 생성 및 학습
        policy_kwargs = dict(
            features_extractor_class=AvoidStoppedObserverExtractor,
            features_extractor_kwargs=dict(features_dim=self.feature_dim),
        )

        # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\AvoidStoppedObserver\logs\direct_mlp.zip"
        # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\AvoidStoppedObserver\logs\pretrained2.zip"
        # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\AvoidStoppedObserver\logs\pretrained3.zip"
        # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\AvoidStoppedObserver\logs\pretrained4.zip"
        model_path = r""
        if os.path.exists(model_path):
            print(f"기존 모델을 불러옵니다: {model_path}")
            model = PPO.load(model_path, env=env, device="cuda")
        else:
            model = PPO(
                "MultiInputPolicy",  # Dict observation space 사용
                env,
                policy_kwargs=policy_kwargs,
                device="cuda",
                verbose=1,
                n_steps=1024,
                batch_size=32,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.01,
                learning_rate=3e-4,
                clip_range=0.2,
                vf_coef=0.5,
                max_grad_norm=0.5,
            )

        # pretrained AE
        # pretrained_ae_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\autoencoder_best.pth"
        pretrained_ae_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\pretrained\avoid_observer\autoencoder_tyndall_log.pth"
        pretrained_ae = torch.load(pretrained_ae_path, map_location="cuda")

        new_state_dict = OrderedDict()
        for k, v in pretrained_ae.items():
            if k.startswith("encoder."):
                new_key = k.replace("encoder.", "", 1)
                new_state_dict[new_key] = v
        print("load resnet encoder")
        print(
            model.policy.features_extractor.resnet.load_state_dict(
                new_state_dict, strict=False
            )
        )

        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

        # 모델 저장
        save_path = os.path.join(self.save_dir, "ppo_find_avoid_observer.zip")
        tmp_path = save_path.replace("zip", "tmp")
        model.save(tmp_path)
        os.replace(tmp_path, save_path)
        print(f"모델 저장 완료: {save_path}")

    def _test(self):
        last_model_path = None
        model = None

        # 커스텀 환경 래핑
        def make_env():
            return EnvAvoidObserver(
                max_steps=self.max_step_length,
                num_observers=50,
                random_start=False,
                move_observer=True,
            )

        env = DummyVecEnv([make_env])
        obs = env.reset()

        # 초기 렌더링을 위해 단일 환경에 접근
        single_env = env.envs[0]
        single_env.render()

        pygame.display.set_caption("FindAvoidObserver Test (ESC: Quit)")
        clock = pygame.time.Clock()

        # 상태 메시지 폰트 초기화
        try:
            font = pygame.font.SysFont("Arial", 24)
        except:
            font = None

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 모델 파일 탐색 및 필요시 reload
            model_path = os.path.join(self.save_dir, "ppo_find_avoid_observer.zip")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(
                        r"ppo_find_avoid_observer_best_(\d+)_\d+\.zip", fname
                    )
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
                try:
                    # env 매개변수와 함께 모델 로드
                    model = PPO.load(model_path, env=env, device="cuda")
                    last_model_path = model_path
                except Exception as e:
                    print(f"모델 로드 실패: {e}")
                    model = None
            elif model is None:
                print("테스트 가능한 모델 파일이 없습니다. (기본 PPO로 테스트)")
                policy_kwargs = dict(
                    features_extractor_class=AvoidStoppedObserverExtractor,
                    features_extractor_kwargs=dict(features_dim=self.feature_dim),
                )
                model = PPO(
                    "MultiInputPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    device="cuda",
                    verbose=1,
                )
                last_model_path = None

            # 모델이 제대로 로드되었을 때만 예측 수행
            if model is not None:
                # 모델 예측
                action, _ = model.predict(obs, deterministic=self.deterministic)
                obs, reward, done, info = env.step(action)
                single_env.render()

                if done[0]:
                    obs = env.reset()
            else:
                # 모델이 없으면 대기 (화면은 계속 렌더링)
                single_env.render()

                # 상태 메시지 표시
                if font is not None and hasattr(single_env, "screen"):
                    text_surface = font.render(
                        "모델 로딩 중... 기다려주세요", True, (255, 255, 0)
                    )
                    single_env.screen.blit(text_surface, (10, 50))
                    pygame.display.flip()

                # 모델 로딩 대기 메시지 표시를 위해 잠시 대기
                time.sleep(0.5)

            clock.tick(self.fps)

        single_env.stop_recording()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    find_avoid_observer = FindAvoidObserver()
    find_avoid_observer.save_dir = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Find_op_map_path"
    )
    find_avoid_observer.log_dir = "logs"
    find_avoid_observer._train()
