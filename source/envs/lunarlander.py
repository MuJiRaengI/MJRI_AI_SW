import os
import sys

sys.path.append(os.path.abspath("."))

import re
import json
import neat
from datetime import datetime

# import gym
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
from source.envs.env import Env
from source.algorithm.ga_lunarlander import GALunarLander


class LunarLander(Env):
    def __init__(self):
        super().__init__()
        self.env_id = "LunarLander-v3"
        self.total_timesteps = 50000000
        self.save_freq = 1000
        self.logging_freq = 1000
        self.n_envs = 8
        self.scale = 4
        self.n_stack = 8
        self.deterministic = False

    def key_info(self) -> str:
        return "해당 게임은 지원하지 않습니다"

    def _self_play(self, *args, **kwargs):
        return

    def _random_play(self, *args, **kwargs):
        return

    def _train(self, *args, **kwargs):
        if self.env_id == "LunarLander-v3":
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\lunarlander\config.json"

        with open(config_path, "r") as f:
            config = json.load(f)

        ga = GALunarLander(save_dir=os.path.join(self.save_dir, "train"))
        ga.learn(config)

        # 학습 완료 신호
        if self.training_queue is not None:
            self.training_queue.put(("done", None))

    def _test(self, *args, **kwargs):
        if self.env_id == "LunarLander-v3":
            config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\lunarlander\config.json"

        with open(config_path, "r") as f:
            config = json.load(f)
        ga = GALunarLander(save_dir=os.path.join(self.save_dir, "test"))

        winner_path = None

        # find winner path
        best_genome_path = os.path.join(self.save_dir, "best_genome.pkl")
        ckpt_path_list = self.sort_checkpoint_files_by_fitness(self.save_dir)
        if os.path.exists(best_genome_path):
            winner_path = best_genome_path
        elif len(ckpt_path_list) > 0:
            winner_path = os.path.join(self.save_dir, ckpt_path_list[0])
        else:
            return

        neat_config_path = config["neat_config_path"]
        neat_config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            neat_config_path,
        )

        ga.load_winner_net(winner_path, neat_config)

        test_episodes = config["test_episodes"]
        test_steps = config["test_steps"]

        env = ga.make_env(render_mode="human")
        for _ in range(test_episodes):
            obs, _ = env.reset()
            for _ in range(test_steps):
                action = ga.predict(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    break
        env.close()

        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def sort_checkpoint_files_by_fitness(self, directory_path, descending=True):
        """fitness 기준으로 checkpoint 파일들을 정렬"""

        # .pkl 파일들 중 checkpoint 파일만 필터링
        pkl_files = [
            f
            for f in os.listdir(directory_path)
            if f.endswith(".pkl") and "fitness_" in f
        ]

        def extract_fitness(filename):
            """파일명에서 fitness 값 추출"""
            # fitness_-83.33 또는 fitness_123.45 패턴 찾기
            match = re.search(r"fitness_(-?\d+\.?\d*)", filename)
            if match:
                return float(match.group(1))
            return float("-inf")  # fitness를 찾을 수 없으면 최저값

        # fitness 기준으로 정렬 (기본: 내림차순 = 높은 점수부터)
        sorted_files = sorted(pkl_files, key=extract_fitness, reverse=descending)

        return sorted_files


if __name__ == "__main__":
    pass
