import os
import sys

sys.path.append(os.path.abspath("."))

import re
import json
import time

# import gym
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import torch
import numpy as np
from collections import deque
import keyboard
import pygame
from stable_baselines3 import PPO
from source.ai.rl.model.breakout import BreakoutResnet
from stable_baselines3.common.env_util import make_vec_env
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from source.ai.rl.BBF_agent.BBF import BBF
from source.ai.rl.model.breakout_bbf import BBF_Model

from source.ai.rl.agent.pqn import PQN
from source.ai.rl.model.atari_cnn import QNetwork


class Breakout(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "Breakout-v4"
        # self.total_timesteps = 50000000
        self.total_timesteps = 500000
        self.save_freq = 1000
        self.logging_freq = 1000
        self.n_envs = 8
        self.scale = 4
        self.n_stack = 8
        self.deterministic = False

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽, SPACE: FIRE(시작/재시작)\n"

    def _self_play(self, *args, **kwargs):
        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        env.metadata["render_fps"] = self.fps
        obs, info = env.reset()

        pygame.init()
        frame = env.render()
        h, w, _ = frame.shape
        h, w = h * self.scale, w * self.scale
        screen = pygame.display.set_mode((h, w))
        pygame.display.set_caption(
            "Breakout Manual Play (A: Left, D: Right, SPACE: Fire)"
        )
        clock = pygame.time.Clock()

        running = True
        action = 0

        while running:
            # 윈도우 종료 처리
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

            # 키보드 입력 처리 (pygame 방식)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                action = 3  # LEFT
            elif keys[pygame.K_d]:
                action = 2  # RIGHT
            elif keys[pygame.K_SPACE]:
                action = 1  # FIRE
            else:
                action = 0  # NOOP

            # 환경 업데이트
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()

            # 화면 출력
            # surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            # screen.blit(surface, (0, 0))
            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            # 에피소드 끝났으면 리셋
            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self, *args, **kwargs):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = self.fps
        obs, info = env.reset()

        pygame.init()
        frame = env.render()
        h, w, _ = frame.shape
        h, w = h * self.scale, w * self.scale
        screen = pygame.display.set_mode((h, w))
        pygame.display.set_caption("Breakout Random Play (Pygame UI)")
        clock = pygame.time.Clock()

        running = True
        while running:
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

            # 랜덤 액션
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()

            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self, *args, **kwargs):
        agent_type = "pqn"
        if agent_type == "bbf":
            log_dir = os.path.join(self.save_dir, self.log_dir)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)

            env = gym.make("ALE/Breakout-v5")
            n_actions = env.action_space.n

            # 진행상황 전달
            if self.training_queue is not None:
                self.training_queue.put(("total_steps", self.total_timesteps))

            # 모델 생성 및 학습
            # model = PPO("CnnPolicy", env, verbose=1, device="cuda", learning_rate=2.5e-4)

            model = BBF_Model(
                n_actions,  # env action space size
                hiddens=2048,  # representation dim
                scale_width=4,  # cnn channel scale
                num_buckets=51,  # buckets in distributional RL
                Vmin=-10,  # min value in distributional RL
                Vmax=10,  # max value in distributional RL
                resize=(96, 72),  # input resize
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
            )

            model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Breakout\logs\ppo_breakout_590000_steps.zip"
            if os.path.exists(model_path):
                print(f"기존 모델을 불러옵니다: {model_path}")
                agent.load(model_path)

            agent.learn(
                total_timesteps=self.total_timesteps,
                save_freq=self.save_freq,
                save_path=self.save_dir,
                name_prefix="bbf_breakout",  # save file name prefix
            )

            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

            # 모델 저장
            save_path = os.path.join(self.save_dir, "bbf_breakout.pth")
            agent.save(save_path)
            print(f"모델 저장 완료: {save_path}")

        elif agent_type == "pqn":
            # config = {
            #     "env_name": self.env_id,
            #     "n_actions": 4,
            #     "device": "cuda:0",
            #     "save_dir": self.save_dir,
            #     "logging_freq": 100,
            #     "detailed_logging_freq": 500,
            #     # model
            #     "model_name": "pqn_breakout",
            #     "max_best_models": 3,
            #     "mid_channels": 32,
            #     "out_channels": 512,
            #     "screen_size": (84, 84),
            #     "norm_type": "layer_norm",
            #     "norm_input": False,
            #     # learning settings
            #     "total_timesteps": self.total_timesteps,
            #     "total_timesteps_decay": self.total_timesteps,
            #     "num_envs": self.n_envs,
            #     "num_steps": 128,
            #     "num_epochs": 1,
            #     "num_minibatches": 4,
            #     "test_during_training": True,
            #     "num_test_envs": 2,
            #     "eps_start": 1.0,
            #     "eps_finish": 0.01,
            #     "eps_decay": 0.1,
            #     "rew_scale": 1.0,
            #     "gamma": 0.99,
            #     "lam": 0.95,
            #     "lr": 2.5e-4,
            #     "lr_linear_decay": True,
            #     "max_grad_norm": 0.5,
            #     "using_async": False,
            #     "frame_stack": 4,
            #     "frame_skip": 4,
            #     "noop_max": 30,
            #     "terminal_on_life_loss": True,
            #     "grayscale_obs": True,
            #     "scale_obs": False,
            #     "seed": None,
            # }

            # json load
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\pqn\breakout\config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # save original config
            save_config_path = os.path.join(
                self.save_dir, config["save_dir"], "config_original.json"
            )
            if not os.path.exists(os.path.dirname(save_config_path)):
                os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
            with open(save_config_path, "w") as f:
                json.dump(config, f, indent=4)

            # update config
            config["save_dir"] = os.path.join(self.save_dir, config["save_dir"])
            config["screen_size"] = tuple(config["screen_size"])

            agent = PQN(
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            model = QNetwork(
                in_channels=config["frame_stack"],
                mid_channels=config["mid_channels"],
                out_channels=config["out_channels"],
                n_actions=config["n_actions"],
                size=config["screen_size"],
                norm_type=config["norm_type"],
                norm_input=config["norm_input"],
            )

            architecture_info = model.body.get_architecture_info()
            agent.logger.info(f" QNetwork-body 모델 구조: {architecture_info}")

            agent.learn(
                model=model,
                config=config,
            )

            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

        else:
            raise ValueError(f"지원하지 않는 에이전트 타입: {agent_type}")

    def _test(self, *args, **kwargs):
        last_model_path = None
        model = None

        env = gym.make("ALE/Breakout-v5", render_mode="rgb_array")
        n_actions = env.action_space.n
        obs, _ = env.reset()

        model = BBF_Model(n_actions).cuda()

        # 학습과 동일하게 FrameStack 적용
        # print(obs.shape)
        state = model.preprocess(obs).unsqueeze(0)
        # print(state.shape)
        states = deque(maxlen=4)
        for i in range(4):
            states.append(state)

        pygame.init()
        last_frame = None
        frame = np.array(env.render())
        h, w, _ = frame.shape
        h, w = h * self.scale, w * self.scale
        screen = pygame.display.set_mode((h, w))
        pygame.display.set_caption("Breakout Test (Pygame UI)")
        clock = pygame.time.Clock()

        running = True
        while running:
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

            # 모델 파일 탐색 및 필요시 reload
            model_path = os.path.join(self.save_dir, "bbf_breakout.pth")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"bbf_breakout_(\d+)_steps\.pth", fname)
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

            # 모델 예측
            # model input : [batch, frameStack, channel, height, width]
            action = model.predict(
                torch.cat(list(states), -3).unsqueeze(0)
            )  # 마지막 4개 Frame 입력
            obs, reward, done, trunc, info = env.step(action.cpu().numpy())
            done = done or trunc
            last_frame = frame
            frame = np.array(env.render())

            if np.array_equal(last_frame, frame):
                env.step(1)

            # FrameStack 적용
            state = model.preprocess(obs).unsqueeze(0)
            states.append(state)

            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            if done:
                obs, _ = env.reset()

                # FrameStack 적용
                state = model.preprocess(obs).unsqueeze(0)
                states = deque(maxlen=4)
                for i in range(4):
                    states.append(state)

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    breakout = Breakout()
    # breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test"
    # breakout.log_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test\logs"
    breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test"
    # breakout._test()
    breakout.play(
        save_dir=breakout.save_dir,
        mode="train",
        queue=None,
    )
