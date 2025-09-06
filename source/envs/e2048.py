import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time

import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import ale_py
import time
import pygame

gym.register_envs(ale_py)
# from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from source.envs.env import Env

from source.envs.env2048 import Env2048PG, _autoplay_step
from source.env_callback.save_on_step_callback import SaveOnStepCallback
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
from source.ai.rl.BBF_agent.BBF import BBF
from source.ai.rl.model.resnet_2048 import ResNet2048Extractor


class E2048(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "Env2048"
        self.total_timesteps = 100000000
        self.save_freq = 10000
        self.logging_freq = 10000
        self.size = 4
        self.max_exp = 16

    def key_info(self) -> str:
        return (
            "[조작법] 0: ↑(Up), 1: →(Right), 2: ↓(Down), 3: ←(Left), r:초기화, q:종료\n"
        )

    def _self_play(self, *args, **kwargs):
        env = Env2048PG(size=self.size)

        done = False
        autoplay = False
        action = None

        running = True
        while running:
            # 최초 렌더
            obs = env.reset()
            env.render()

            while not done:
                # render_queue로부터 stop 신호를 받으면 중단
                if self.render_queue is not None and not self.render_queue.empty():
                    msg = self.render_queue.get()
                    if isinstance(msg, tuple) and msg[0] == "stop":
                        running = False
                        break

                event = pygame.event.wait()
                if event.type == pygame.QUIT:
                    running = False
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return
                else:
                    if event.type == pygame.KEYDOWN:
                        if event.key in (
                            pygame.K_UP,
                            pygame.K_RIGHT,
                            pygame.K_DOWN,
                            pygame.K_LEFT,
                        ):
                            key2act = {
                                pygame.K_UP: 0,
                                pygame.K_RIGHT: 1,
                                pygame.K_DOWN: 2,
                                pygame.K_LEFT: 3,
                            }
                            action = key2act[event.key]
                        elif event.key == pygame.K_r:
                            print("Resetting environment.")
                            obs = env.reset()
                            done = False
                            action = None

                        # 스텝
                        if action is not None and not done:
                            obs, reward, terminated, truncated, info = env.step(action)
                            action = None

                        env.render()

        env.stop_recording()
        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self, *args, **kwargs):
        env = Env2048PG(size=self.size)
        obs = env.reset()
        env.render()

        action = None
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

            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # pygame.display.flip()

            if terminated or truncated:
                obs, info = env.reset()

        env.stop_recording()

        env.close()

    def _train(self, *args, **kwargs):
        log_dir = os.path.join(self.save_dir, self.log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        env = Env2048PG(
            size=self.size,
            max_exp=self.max_exp,
            reward_mode="custom",
            invalid_move_penalty=-1,
        )

        # 진행상황 전달
        if self.training_queue is not None:
            self.training_queue.put(("total_steps", self.total_timesteps))

        model = ResNet2048Extractor(
            features_dim=256, size=self.size, max_exp=self.max_exp
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
            weight_decay=0.1,  # weight decay in optimizer
        )

        agent.learn(
            total_timesteps=self.total_timesteps,
            save_freq=self.save_freq,
            save_path=self.save_dir,
            name_prefix="bbf_2048",
        )

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

        # 모델 저장
        save_path = os.path.join(self.save_dir, "bbf_e2048.zip")
        tmp_path = save_path.replace("zip", "tmp")
        model.save(tmp_path)
        os.replace(tmp_path, save_path)
        print(f"모델 저장 완료: {save_path}")

    def _test(self, *args, **kwargs):
        last_model_path = None
        model = None
        env = Env2048PG(size=self.size)
        obs, info = env.reset()
        env.render()
        # clock = pygame.time.Clock()
        while True:
            # 모델 파일 탐색 및 필요시 reload
            model_path = os.path.join(self.save_dir, "ppo_e2048.zip")
            if not os.path.exists(model_path):
                max_steps = -1
                max_steps_path = None
                for fname in os.listdir(self.save_dir):
                    match = re.match(r"ppo_e2048_best_(\d+)_([-\d\.]+)\.zip", fname)
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
                model.policy.eval()
                last_model_path = model_path
            elif model is None:
                print("테스트 가능한 모델 파일이 없습니다. (기본 PPO로 테스트)")
                model = PPO("CnnPolicy", env, device="cpu")
                model.policy.eval()
                last_model_path = None

            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # action, _ = model.predict(obs, deterministic=False)
            action_mask = obs["vector"]
            obs_tensor = torch.as_tensor(obs["board"]).to(model.device)
            if len(obs_tensor.shape) == 3:
                obs_tensor = obs_tensor.unsqueeze(0)  # (1, C, H, W) 형태로 맞추기
            with torch.no_grad():
                dist = model.policy.get_distribution(obs_tensor)
                action_probs = dist.distribution.probs.cpu().numpy().squeeze()
                action_probs *= action_mask  # action mask 적용
                action = np.argmax(action_probs)

            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            # time.sleep(0.1)
            pygame.event.get()
            if terminated or truncated:
                while True:
                    event = pygame.event.wait()
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_r:
                            done = False
                            action = None
                            break
                    time.sleep(0.1)  # 잠시 대기
                obs, info = env.reset()
                # 에피소드가 끝나도 env는 유지, 모델만 reload
                continue
            # clock.tick(getattr(self, "fps", 10))
        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


class BoardToImageWrapper(gym.ObservationWrapper):
    """
    (H,W) int32 보드 -> (1,H,W) float32 in [0,1]
    값은 log2 정규화: 0->0, 2->1, 4->2 ... 를 max_exp로 나눔
    """

    def __init__(self, env, max_exp=15, dtype=np.float32):
        super().__init__(env)
        self.max_exp = int(max_exp)
        # Dict observation space에서 "board" 부분의 shape 가져오기
        c, h, w = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.max_exp + 1, h, w), dtype=dtype
        )
        self.dtype = dtype

    # def observation(self, obs):
    #     # obs = obs[0]
    #     obs = obs["board"][0]
    #     exp = np.zeros_like(obs, dtype=self.dtype)
    #     nz = obs > 0
    #     exp[nz] = np.log2(obs[nz]).astype(self.dtype)
    #     exp = np.clip(exp, 0, self.max_exp) / float(self.max_exp)
    #     return exp[None, ...]  # (1,H,W)

    def observation(self, obs):
        # obs: (4, 4) int32, 각 칸의 값 (0, 2, 4, ..., 2**15)
        obs = obs["board"][0]
        exps = np.zeros_like(obs, dtype=np.int32)
        nz = obs > 0
        exps[nz] = np.log2(obs[nz]).astype(np.int32)
        # one-hot 인코딩: (4, 4) -> (4, 4, 16)
        onehot = np.eye(self.max_exp + 1, dtype=np.float32)[exps]  # (4, 4, 16)
        onehot = np.transpose(onehot, (2, 0, 1))  # (16, 4, 4)
        return onehot


def test():
    env = E2048()
    env.play(
        mode="train",
        render_queue=None,
        training_queue=None,
        save_dir="./test",
        log_dir="./test",
    )


def time_check():
    env = Env2048PG(size=4)
    _ = env.reset()
    tmr = time.time()
    count = 0
    while (time.time() - tmr) < 1:
        _, _, done, _, _ = env.step(0)
        count += 1

    print(f"1초 동안 {count} 에피소드 진행")  # 1361


if __name__ == "__main__":
    time_check()
