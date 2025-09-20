import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time
import json
import pygame
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from source.envs.env import Env
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from source.ai.rl.model.find_avoid_observer_model import FindAvoidObserverExtractor
from ..envs.env_avoid_observber import EnvAvoidObserver
from source.ai.rl.BBF_agent.BBF import BBF
from source.ai.rl.model.avoid_observer_bbf import BBF_Model


class FindAvoidObserver(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "FindAvoidObserver-v1"
        self.total_timesteps = 100000
        self.save_freq = 10000
        self.logging_freq = 10000
        self.feature_dim = 256
        self.n_envs = 4
        self.scale = 1  # 스케일링 없음 (이미 적절한 크기)
        self.deterministic = False

    def key_info(self) -> str:
        return "[조작법] D(→), C(↘), S(↓), Z(↙), A(←), Q(↖), W(↑), E(↗)\n" "ESC: 종료\n"

    def _self_play(self, *args, **kwargs):
        env = EnvAvoidObserver()
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

    def _random_play(self, *args, **kwargs):
        env = EnvAvoidObserver()
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

    def _train(self, *args, **kwargs):
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # 커스텀 환경 래핑
        def make_env():
            return EnvAvoidObserver()

        env = make_env()
        n_actions = env.action_space.n

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        model = BBF_Model(
            n_actions,  # env action space size
            hiddens=2048,  # representation dim
            scale_width=4,  # cnn channel scale
            num_buckets=51,  # buckets in distributional RL
            Vmin=-10,  # min value in distributional RL
            Vmax=10,  # max value in distributional RL
            resize=(80, 160),  # input resize
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
            gym_env=True,
            stackFrame=False,
        )

        # model_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\Find_op_map_path\logs\ppo_find_avoid_observer_best_44921_3.zip"
        # if os.path.exists(model_path):
        #     print(f"기존 모델을 불러옵니다: {model_path}")
        #     model = PPO.load(model_path, env=env, device="cuda")

        agent.learn(
            total_timesteps=self.total_timesteps,
            save_freq=self.save_freq,
            save_path=self.save_dir,
            name_prefix="bbf_find_avoid_observer",  # save file name prefix
            project_name="find_avoid_observer",
            exp_name="BBF",
        )

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

        # 모델 저장
        save_path = os.path.join(self.save_dir, "bbf_find_avoid_observer.pth")
        agent.save(save_path)
        print(f"모델 저장 완료: {save_path}")

    def _test(self, *args, **kwargs):
        last_model_path = None
        model = None

        # 커스텀 환경 래핑
        def make_env():
            return EnvAvoidObserver()

        env = make_env()
        n_actions = env.action_space.n
        obs = env.reset()

        model = BBF_Model(n_actions, resize=(80, 160)).cuda()  # input resize
        state = model.preprocess(obs).unsqueeze(0)

        # 초기 렌더링을 위해 단일 환경에 접근
        single_env = env
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
            model_path = os.path.join(self.save_dir, "bbf_find_avoid_observer.zip")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"bbf_find_avoid_observer_(\d+)_steps\.pth", fname)
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
                action = model.predict(state.unsqueeze(0))
                obs, reward, done, info = env.step(action.cpu().item())
                single_env.render()

                if done:
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
