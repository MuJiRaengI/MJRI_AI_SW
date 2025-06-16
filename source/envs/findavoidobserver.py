import os
import pygame
import numpy as np
import PySide6.QtWidgets as QtWidgets
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import torch
import time
import json

from source.ai.rl.model.find_avoid_observer_model import FindAvoidObserverPolicy
from .env_avoid_observber import EnvAvoidObserver


class FindAvoidObserver:
    def __init__(self):
        self.env = None

    def reset(self):
        self.env = EnvAvoidObserver()
        return self.env.reset()

    def play(self, solution_dir, mode):
        if self.env is None:
            self.reset()
        if mode == "self_play":
            self._self_play()
        elif mode == "random_play":
            self._random_play()
        elif mode == "train":
            self._train(solution_dir)
        elif mode == "test":
            self._test(solution_dir)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def key_info(self) -> str:
        return (
            "[조작법] 방향키: D(→), C(↘), S(↓), Z(↙), A(←), Q(↖), W(↑), E(↗)\nESC: 종료\n"
            "키보드로 조작하거나, ESC를 눌러 종료할 수 있습니다."
        )

    def _self_play(self):
        env = self.env
        running = True
        while running:
            obs = env.reset()
            done = False
            action = None
            env.render()
            while not done and running:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            done = True
                        elif event.key == pygame.K_d:
                            action = 0
                        elif event.key == pygame.K_c:
                            action = 1
                        elif event.key == pygame.K_s:
                            action = 2
                        elif event.key == pygame.K_z:
                            action = 3
                        elif event.key == pygame.K_a:
                            action = 4
                        elif event.key == pygame.K_q:
                            action = 5
                        elif event.key == pygame.K_w:
                            action = 6
                        elif event.key == pygame.K_e:
                            action = 7
                obs, reward, done, _ = env.step(action)
                env.render()
        env.stop_recording()
        pygame.quit()

    def _random_play(self):
        env = self.env
        running = True
        while running:
            obs = env.reset()
            done = False
            env.render()
            while not done and running:
                action = env.action_space.sample()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        done = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                            done = True
                obs, reward, done, _ = env.step(action)
                env.render()
        env.stop_recording()
        pygame.quit()

    def _train(self, solution_dir):
        self.reset()
        log_dir = os.path.join(solution_dir, "logs")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "ppo_find_avoid_observer_train_log.json")
        save_path = os.path.join(solution_dir, "ppo_find_avoid_observer.zip")
        model = PPO(FindAvoidObserverPolicy, self.env, verbose=1)
        total_timesteps = 1000000  # 원하는 학습 스텝 수
        model.learn(total_timesteps=total_timesteps)
        model.save(save_path)

        episode_rewards = []
        episode_lengths = []
        obs, info = self.env.reset(), {}
        timestep = 0
        last_print = 0
        start_time = time.time()
        while timestep < total_timesteps:
            done = False
            ep_reward = 0
            ep_length = 0
            while not done and timestep < total_timesteps:
                action, _ = model.predict(obs, deterministic=False)
                step_result = self.env.step(int(action))
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    obs, reward, done, info = step_result
                self.env.render()
                ep_reward += reward
                ep_length += 1
                timestep += 1
                if timestep % 1000 == 0:
                    model.save(save_path)
                    progress = timestep / total_timesteps
                    elapsed = time.time() - start_time
                    safe_episode_rewards = [float(r) for r in episode_rewards]
                    safe_episode_lengths = [int(l) for l in episode_lengths]
                    with open(log_path, "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "episode_rewards": safe_episode_rewards,
                                "episode_lengths": safe_episode_lengths,
                                "timestep": int(timestep),
                                "progress": float(progress),
                                "elapsed_seconds": float(elapsed),
                            },
                            f,
                            ensure_ascii=False,
                            indent=2,
                        )
                    print(f"중간 로그가 저장되었습니다: {log_path}")
                if timestep - last_print >= 1000:
                    elapsed = time.time() - start_time
                    percent = 100 * timestep / total_timesteps
                    print(
                        f"[Train] Step: {timestep}/{total_timesteps} | Episodes: {len(episode_rewards)} | Last reward: {ep_reward} | Elapsed: {elapsed:.1f}s | Progress: {percent:.1f}%"
                    )
                    last_print = timestep
            episode_rewards.append(ep_reward)
            episode_lengths.append(ep_length)
            obs, info = self.env.reset(), {}
        model.save(save_path)
        progress = timestep / total_timesteps
        elapsed = time.time() - start_time
        safe_episode_rewards = [float(r) for r in episode_rewards]
        safe_episode_lengths = [int(l) for l in episode_lengths]
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "episode_rewards": safe_episode_rewards,
                    "episode_lengths": safe_episode_lengths,
                    "timestep": int(timestep),
                    "progress": float(progress),
                    "elapsed_seconds": float(elapsed),
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
        print(f"모델이 저장되었습니다: {save_path}")
        print(f"학습 로그가 저장되었습니다: {log_path}")
        self.env.close()

    def _test(self, solution_dir):
        pass
