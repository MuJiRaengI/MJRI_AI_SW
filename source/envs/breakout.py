import os
import sys

sys.path.append(os.path.abspath("."))

import re
import json
import time
from datetime import datetime

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

from source.ai.rl.model.pqn_enhance import QNetwork
from source.ai.rl.model.hadamax import QNetwork, QNetwork_Impala


class Breakout(Env):
    def __init__(self):
        super().__init__()
        # self.env_id = "Breakout-v4"
        self.env_id = "ALE/Breakout-v5"
        self.total_timesteps = 50000000
        # self.total_timesteps = 500000
        self.save_freq = 1000
        self.logging_freq = 1000
        self.n_envs = 8
        self.scale = 4
        self.n_stack = 8
        self.deterministic = False

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽, SPACE: FIRE(시작/재시작)\n"

    def _self_play(self, *args, **kwargs):
        env = gym.make(self.env_id, render_mode="rgb_array")
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
        agent_type = "hadamax"
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
            # json load
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\pqn\breakout\config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # save original config
            now = datetime.now()
            now_time = now.strftime("%Yy_%mm_%dd_%Hh%Mm%Ss")
            save_config_path = os.path.join(
                self.save_dir, config["save_dir"], now_time, "config_original.json"
            )
            if not os.path.exists(os.path.dirname(save_config_path)):
                os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
            with open(save_config_path, "w") as f:
                json.dump(config, f, indent=4)

            # update config
            config["save_dir"] = os.path.join(
                self.save_dir, config["save_dir"], now_time
            )
            config["screen_size"] = tuple(config["screen_size"])

            model = QNetwork(
                in_channels=config["frame_stack"],
                base_channels=config["base_channels"],
                out_channels=config["out_channels"],
                n_actions=config["n_actions"],
                size=config["screen_size"],
                norm_type=config["norm_type"],
                norm_input=config["norm_input"],
            )

            if config["pre_trained"]:
                pre_trained_path = config["pre_trained_path"]
                if os.path.exists(pre_trained_path):
                    print(f"Pre-trained 모델 로드: {pre_trained_path}")
                    model.load_state_dict(
                        torch.load(pre_trained_path, map_location=config["device"])
                    )
                else:
                    print("Pre-trained 모델 파일을 찾을 수 없습니다.")

            agent = PQN(
                model=model,
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            agent.learn(config=config)

            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

        elif agent_type == "hadamax":
            from source.ai.rl.model.hadamax import QNetwork

            # json load
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\hadamax\breakout\config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # save original config
            now = datetime.now()
            now_time = now.strftime("%Yy_%mm_%dd_%Hh%Mm%Ss")
            save_config_path = os.path.join(
                self.save_dir, config["save_dir"], now_time, "config_original.json"
            )
            if not os.path.exists(os.path.dirname(save_config_path)):
                os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
            with open(save_config_path, "w") as f:
                json.dump(config, f, indent=4)

            # update config
            config["save_dir"] = os.path.join(
                self.save_dir, config["save_dir"], now_time
            )
            config["screen_size"] = tuple(config["screen_size"])

            model = QNetwork(
                encoder_type=config["encoder_type"],
                action_dim=config["n_actions"],
                in_channels=config["frame_stack"],
                norm_type=config["norm_type"],
                norm_input=config["norm_input"],
                input_size=config["screen_size"],
                channels=config["channels"],
            )

            if config["pre_trained"]:
                pre_trained_path = config["pre_trained"]
                if os.path.exists(pre_trained_path):
                    print(f"Pre-trained 모델 로드: {pre_trained_path}")
                    model.load_state_dict(
                        torch.load(pre_trained_path, map_location=config["device"])
                    )
                else:
                    print("Pre-trained 모델 파일을 찾을 수 없습니다.")

            agent = PQN(
                model=model,
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            agent.learn(config=config)

            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

        elif agent_type == "hadamax_convnext":
            from source.ai.rl.model.hadamax_convnext import QNetwork

            # json load
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\hadamax\breakout\config_convnext.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # save original config
            now = datetime.now()
            now_time = now.strftime("%Yy_%mm_%dd_%Hh%Mm%Ss")
            save_config_path = os.path.join(
                self.save_dir, config["save_dir"], now_time, "config_original.json"
            )
            if not os.path.exists(os.path.dirname(save_config_path)):
                os.makedirs(os.path.dirname(save_config_path), exist_ok=True)
            with open(save_config_path, "w") as f:
                json.dump(config, f, indent=4)

            # update config
            config["save_dir"] = os.path.join(
                self.save_dir, config["save_dir"], now_time
            )
            config["screen_size"] = tuple(config["screen_size"])

            model = QNetwork(
                action_dim=config["n_actions"],
                in_channels=config["frame_stack"],
                norm_input=config["norm_input"],
                input_size=config["screen_size"],
                hidden_dim=config["hidden_dim"],
            )

            if config["pre_trained"]:
                pre_trained_path = config["pre_trained"]
                if os.path.exists(pre_trained_path):
                    print(f"Pre-trained 모델 로드: {pre_trained_path}")
                    model.load_state_dict(
                        torch.load(pre_trained_path, map_location=config["device"])
                    )
                else:
                    print("Pre-trained 모델 파일을 찾을 수 없습니다.")

            agent = PQN(
                model=model,
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            agent.learn(config=config)

            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

        else:
            raise ValueError(f"지원하지 않는 에이전트 타입: {agent_type}")

    def _test(self, *args, **kwargs):
        agent_type = "hadamax"
        if agent_type == "bbf":
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

        elif agent_type == "pqn":

            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Breakout\results\pqn\2025y_08m_23d_16h46m50s\config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # update config
            config["save_dir"] = os.path.join(config["save_dir"], "test")
            config["screen_size"] = tuple(config["screen_size"])

            model = QNetwork(
                in_channels=config["frame_stack"],
                base_channels=config["base_channels"],
                out_channels=config["out_channels"],
                n_actions=config["n_actions"],
                size=config["screen_size"],
                norm_type=config["norm_type"],
                norm_input=config["norm_input"],
            ).to(config["device"])

            agent = PQN(
                model=model,
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            # 1. 환경 생성
            env = agent.make_env(
                env_id=self.env_id,
                render_mode="rgb_array",
                frame_stack=config["frame_stack"],
                frame_skip=config["frame_skip"],
                noop_max=config["noop_max"],
                screen_size=config["screen_size"],
                terminal_on_life_loss=config["terminal_on_life_loss"],
                grayscale_obs=config["grayscale_obs"],
                scale_obs=config["scale_obs"],
            )()

            # 2. 최신 best 모델 찾기 및 로드
            last_model_path = None
            model_files = []
            for fname in os.listdir(os.path.dirname(config["save_dir"])):
                if (
                    fname.startswith("pqn_")
                    and "_best_" in fname
                    and fname.endswith(".pth")
                ):
                    model_files.append(fname)

            if model_files:
                # 가장 높은 reward를 가진 모델 선택
                best_model = None
                best_reward = -float("inf")
                for fname in model_files:
                    try:
                        # 파일명에서 reward 추출 (언더바로 변경된 형태)
                        reward_part = fname.split("_reward_")[1].replace(".pth", "")
                        reward = float(reward_part.replace("_", "."))
                        if reward > best_reward:
                            best_reward = reward
                            best_model = fname
                    except:
                        continue

                if best_model:
                    model_path = os.path.join(
                        config["save_dir"].replace("test", ""), best_model
                    )
                    if os.path.exists(model_path):
                        agent.load_model(model_path)
                        last_model_path = model_path
                        print(f"Best 모델 로드: {best_model} (보상: {best_reward:.3f})")
                    else:
                        print("Best 모델 파일을 찾을 수 없습니다.")
            else:
                print("학습된 모델이 없습니다.")

            # 3. Pygame 초기화 및 게임 루프
            pygame.init()
            obs, _ = env.reset()

            # 첫 번째 프레임 렌더링을 위해 env.render() 호출
            # AtariPreprocessing이 grayscale_obs=False로 설정되어 있어야 함
            frame = np.array(env.render())
            h, w, _ = frame.shape
            h, w = h * self.scale, w * self.scale
            screen = pygame.display.set_mode((h, w))
            pygame.display.set_caption("Breakout PQN Test")
            clock = pygame.time.Clock()

            # 4. 테스트 루프
            episode = 0
            total_reward = 0
            episode_length = 0
            running = True

            print(
                f"PQN 테스트 시작! (모델: {last_model_path if last_model_path else 'Random'})"
            )

            while running:
                # 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                if self.render_queue is not None and not self.render_queue.empty():
                    msg = self.render_queue.get()
                    if isinstance(msg, tuple) and msg[0] == "stop":
                        break

                # 5. 행동 예측
                if last_model_path:
                    action, q_values = agent.predict(obs, deterministic=True)
                    # Q-values 출력 (너무 많이 출력되지 않도록 가끔씩만)
                    if episode_length % 30 == 0:  # 30스텝마다 출력
                        print(f"Action: {action}, Q-values: {q_values}")
                else:
                    action = env.action_space.sample()  # 랜덤 행동

                # 6. 환경 스텝
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                episode_length += 1

                # 7. 화면 렌더링
                frame = np.array(env.render())
                scaled_frame = pygame.transform.scale(
                    pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                    (h, w),
                )
                screen.blit(scaled_frame, (0, 0))

                # 8. 텍스트 정보 표시
                font = pygame.font.Font(None, 36)
                text_lines = [
                    f"Episode: {episode}",
                    f"Reward: {total_reward:.1f}",
                    f"Length: {episode_length}",
                    f"Action: {action}",
                    f"Model: {'Best' if last_model_path else 'Random'}",
                ]

                for i, line in enumerate(text_lines):
                    text = font.render(line, True, (255, 255, 255))
                    screen.blit(text, (10, 10 + i * 30))

                pygame.display.flip()

                # 9. 에피소드 종료 처리
                if done or truncated:
                    print(
                        f"Episode {episode} 완료: 보상={total_reward:.1f}, 길이={episode_length}"
                    )

                    episode += 1
                    total_reward = 0
                    episode_length = 0

                    obs, _ = env.reset()
                    time.sleep(1)  # 잠시 대기

                clock.tick(self.fps)

            # 10. 정리
            env.close()
            pygame.quit()
            if self.render_queue is not None:
                self.render_queue.put(("done", None))

        elif agent_type == "hadamax":
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Breakout\results\hadamax\2025y_08m_24d_00h30m58s\config.json"
            with open(config_path, "r") as f:
                config = json.load(f)

            # update config
            config["save_dir"] = os.path.join(config["save_dir"], "test")
            config["screen_size"] = tuple(config["screen_size"])

            model = QNetwork(
                encoder_type=config["encoder_type"],
                action_dim=config["n_actions"],
                in_channels=config["frame_stack"],
                norm_type=config["norm_type"],
                norm_input=config["norm_input"],
                input_size=config["screen_size"],
                channels=config["channels"],
            ).to(config["device"])

            agent = PQN(
                model=model,
                save_dir=config["save_dir"],
                logging_freq=config["logging_freq"],
                detailed_logging_freq=config["detailed_logging_freq"],
            )

            # 1. 환경 생성
            env = agent.make_env(
                env_id=self.env_id,
                render_mode="rgb_array",
                frame_stack=config["frame_stack"],
                frame_skip=config["frame_skip"],
                noop_max=config["noop_max"],
                screen_size=config["screen_size"],
                terminal_on_life_loss=config["terminal_on_life_loss"],
                grayscale_obs=config["grayscale_obs"],
                scale_obs=config["scale_obs"],
            )()

            # 2. 최신 best 모델 찾기 및 로드
            last_model_path = None
            model_files = []
            for fname in os.listdir(os.path.dirname(config["save_dir"])):
                if (
                    fname.startswith("hadamax_")
                    and "_best_" in fname
                    and fname.endswith(".pth")
                ):
                    model_files.append(fname)

            if model_files:
                # 가장 높은 reward를 가진 모델 선택
                best_model = None
                best_reward = -float("inf")
                for fname in model_files:
                    try:
                        # 파일명에서 reward 추출 (언더바로 변경된 형태)
                        reward_part = fname.split("_reward_")[1].replace(".pth", "")
                        reward = float(reward_part.replace("_", "."))
                        if reward > best_reward:
                            best_reward = reward
                            best_model = fname
                    except:
                        continue

                if best_model:
                    model_path = os.path.join(
                        config["save_dir"].replace("test", ""), best_model
                    )
                    if os.path.exists(model_path):
                        agent.load_model(model_path)
                        last_model_path = model_path
                        print(f"Best 모델 로드: {best_model} (보상: {best_reward:.3f})")
                    else:
                        print("Best 모델 파일을 찾을 수 없습니다.")
            else:
                print("학습된 모델이 없습니다.")

            # 3. Pygame 초기화 및 게임 루프
            pygame.init()
            obs, _ = env.reset()

            # 첫 번째 프레임 렌더링을 위해 env.render() 호출
            # AtariPreprocessing이 grayscale_obs=False로 설정되어 있어야 함
            frame = np.array(env.render())
            h, w, _ = frame.shape
            h, w = h * self.scale, w * self.scale
            screen = pygame.display.set_mode((h, w))
            pygame.display.set_caption("Breakout PQN Test")
            clock = pygame.time.Clock()

            # 4. 테스트 루프
            episode = 0
            total_reward = 0
            episode_length = 0
            running = True

            print(
                f"PQN 테스트 시작! (모델: {last_model_path if last_model_path else 'Random'})"
            )

            while running:
                # 이벤트 처리
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                if self.render_queue is not None and not self.render_queue.empty():
                    msg = self.render_queue.get()
                    if isinstance(msg, tuple) and msg[0] == "stop":
                        break

                # 5. 행동 예측
                if last_model_path:
                    action, q_values = agent.predict(obs, deterministic=True)
                    # Q-values 출력 (너무 많이 출력되지 않도록 가끔씩만)
                    if episode_length % 30 == 0:  # 30스텝마다 출력
                        print(f"Action: {action}, Q-values: {q_values}")
                else:
                    action = env.action_space.sample()  # 랜덤 행동

                # 6. 환경 스텝
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                episode_length += 1

                # 7. 화면 렌더링
                frame = np.array(env.render())
                scaled_frame = pygame.transform.scale(
                    pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                    (h, w),
                )
                screen.blit(scaled_frame, (0, 0))

                # 8. 텍스트 정보 표시
                font = pygame.font.Font(None, 36)
                text_lines = [
                    f"Episode: {episode}",
                    f"Reward: {total_reward:.1f}",
                    f"Length: {episode_length}",
                    f"Action: {action}",
                    f"Model: {'Best' if last_model_path else 'Random'}",
                ]

                for i, line in enumerate(text_lines):
                    text = font.render(line, True, (255, 255, 255))
                    screen.blit(text, (10, 10 + i * 30))

                pygame.display.flip()

                # 9. 에피소드 종료 처리
                if done or truncated:
                    print(
                        f"Episode {episode} 완료: 보상={total_reward:.1f}, 길이={episode_length}"
                    )

                    episode += 1
                    total_reward = 0
                    episode_length = 0

                    obs, _ = env.reset()
                    time.sleep(1)  # 잠시 대기

                clock.tick(self.fps)

            # 10. 정리
            env.close()
            pygame.quit()
            if self.render_queue is not None:
                self.render_queue.put(("done", None))
        else:
            raise ValueError(f"지원하지 않는 에이전트 타입: {agent_type}")


if __name__ == "__main__":
    breakout = Breakout()
    # breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test"
    # breakout.log_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test\logs"
    # breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test"
    breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Breakout"
    # breakout._test()
    breakout.play(
        save_dir=breakout.save_dir,
        mode="test",
        queue=None,
    )
