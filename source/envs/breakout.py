import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time
import gym
import keyboard
import pygame
from stable_baselines3 import PPO
from source.ai.rl.model.breakout import BreakoutResnet
from stable_baselines3.common.env_util import make_vec_env
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


class Breakout(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "Breakout-v4"
        self.total_timesteps = 50000000
        # self.save_freq = 5000
        self.save_freq = 1000
        self.logging_freq = 1000
        self.n_envs = 8
        self.scale = 4
        self.n_stack = 8

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽, SPACE: FIRE(시작/재시작)\n"

    def _self_play(self):
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

    def _random_play(self):
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

    def _train(self):
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        env = make_vec_env(self.env_id, n_envs=self.n_envs)
        env = VecFrameStack(env, n_stack=self.n_stack)

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        callback = SaveOnStepCallback(
            save_freq=self.save_freq,
            logging_freq=self.logging_freq,
            save_dir=self.save_dir,
            name_prefix="ppo_breakout",
            log_dir=log_dir,
            progress_queue=self.training_queue,
            verbose=1,
        )
        # 모델 생성 및 학습
        # model = PPO("CnnPolicy", env, verbose=1, device="cuda", learning_rate=2.5e-4)

        policy_kwargs = dict(
            features_extractor_class=BreakoutResnet,
            features_extractor_kwargs=dict(features_dim=64),
        )

        model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Breakout\logs\ppo_breakout_460000_steps.zip"
        if os.path.exists(model_path):
            print(f"기존 모델을 불러옵니다: {model_path}")
            model = PPO.load(model_path, env=env, policy=BreakoutResnet, device="cuda")
        else:
            model = PPO(
                "CnnPolicy",
                env,
                policy_kwargs=policy_kwargs,
                device="cuda",
                verbose=1,
                n_steps=128,  # rollout buffer size (default: 2048, Atari에서는 128~256 추천)
                batch_size=256,  # minibatch size (default: 64, Atari에서는 256~1024 추천)
                n_epochs=4,  # update epochs (default: 10, Atari에서는 4~6 추천)
                gamma=0.99,  # discount factor
                gae_lambda=0.95,  # GAE lambda
                ent_coef=0.01,  # entropy coefficient (default: 0.0, exploration 증가)
                learning_rate=2.5e-4,  # learning rate (Atari 논문/Stable-Baselines3 권장)
                clip_range=0.1,  # policy clip range (default: 0.2, Atari에서는 0.1~0.2)
                vf_coef=0.5,  # value function loss coefficient
                max_grad_norm=0.5,  # gradient clipping
            )
        model.learn(total_timesteps=self.total_timesteps, callback=callback)

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

        # 모델 저장
        save_path = os.path.join(self.save_dir, "ppo_breakout.zip")
        tmp_path = save_path.replace("zip", "tmp")
        model.save(tmp_path)
        os.replace(tmp_path, save_path)
        print(f"모델 저장 완료: {save_path}")

    def _test(self):
        last_model_path = None
        model = None
        # 학습과 동일하게 VecFrameStack 적용
        env = DummyVecEnv([lambda: gym.make(self.env_id, render_mode="rgb_array")])
        env = VecFrameStack(env, n_stack=self.n_stack)
        obs = env.reset()

        pygame.init()
        frame = env.envs[0].render()
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
            model_path = os.path.join(self.save_dir, "ppo_breakout.zip")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"ppo_breakout_best_(\d+)_\d+\.zip", fname)
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
                model = PPO.load(model_path, policy=BreakoutResnet, device="cuda")
                last_model_path = model_path
            elif model is None:
                print("테스트 가능한 모델 파일이 없습니다. (기본 PPO로 테스트)")
                policy_kwargs = dict(
                    features_extractor_class=BreakoutResnet,
                    features_extractor_kwargs=dict(features_dim=64),
                )
                model = PPO(
                    "CnnPolicy",
                    env,
                    policy_kwargs=policy_kwargs,
                    device="cuda",
                    verbose=1,
                )
                last_model_path = None

            # 모델 예측
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            frame = env.envs[0].render()

            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            if done[0]:
                obs = env.reset()

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    breakout = Breakout()
    breakout.save_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test"
    breakout.log_dir = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\test\logs"
    breakout._train()
