import os
import sys

sys.path.append(os.path.abspath("."))
import json
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import numpy as np

from source.algorithm.genetic_algorithm.ga import GeneticAlgorithm
from source.algorithm.genetic_algorithm.network_reporter import NetworkReporter
from source.algorithm.genetic_algorithm.save_best_reporter import SaveBestReporter
from source.algorithm.genetic_algorithm.utils import *
from source.envs.env_2048_v2 import Batch2048EnvFast


class GA2048(GeneticAlgorithm):
    def __init__(
        self,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)

    def make_env(self, render_mode=None):
        return Batch2048EnvFast(num_envs=1)

    def get_reporters(self, config):
        node_names = {
            -1: "Cell[0,0]",
            -2: "Cell[0,1]",
            -3: "Cell[0,2]",
            -4: "Cell[0,3]",
            -5: "Cell[1,0]",
            -6: "Cell[1,1]",
            -7: "Cell[1,2]",
            -8: "Cell[1,3]",
            -9: "Cell[2,0]",
            -10: "Cell[2,1]",
            -11: "Cell[2,2]",
            -12: "Cell[2,3]",
            -13: "Cell[3,0]",
            -14: "Cell[3,1]",
            -15: "Cell[3,2]",
            -16: "Cell[3,3]",
            0: "Left",
            1: "Right",
            2: "Up",
            3: "Down",
        }
        reporters = []
        reporters.append(
            NetworkReporter(config, os.path.join(self.save_dir, "networks"), node_names)
        )
        reporters.append(
            SaveBestReporter(
                save_dir=os.path.join(self.save_dir, "best_genomes"),
                top_n=5,
                config=config,
            )
        )
        return reporters

    def stats_visualize(self, state, save_dir: str):
        save_plot_stats(state, save_dir, filename="fitness_stats.png")
        save_plot_species(state, save_dir, filename="species_stats.png")

    def preprocess_log_flatten(self, obs):
        """
        (N, 4) uint16 → (N, 16) float32
        로그 값 그대로 사용 (이미 log2 저장됨)
        """
        batch_size = obs.shape[0]
        boards_flat = np.zeros((batch_size, 16), dtype=np.float32)

        for i in range(4):  # 각 행
            for j in range(4):  # 각 셀
                shift = (3 - j) * 4
                cell_values = (obs[:, i] >> shift) & 0xF
                boards_flat[:, i * 4 + j] = cell_values.astype(np.float32) / 15

        return boards_flat

    # --------------------------
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            env = self.make_env()

            fitness = 0
            for episode in range(self.num_episodes):
                observation, _ = env.reset()
                observation = self.preprocess_log_flatten(observation)
                episode_reward = 0

                for step in range(self.max_step):
                    output = net.activate(observation[0])
                    action = output.index(max(output))
                    observation, reward, terminated, truncated, info = env.step(action)
                    observation = self.preprocess_log_flatten(observation)
                    board = np.pow(2, observation * 15)
                    # episode_reward += reward

                    if terminated or truncated:
                        break
                # fitness += episode_reward.sum()
                fitness += board.max()

            env.close()
            genome.fitness = float(fitness / self.num_episodes)

    def test_winner(self, genome, config, episodes: int, steps: int):
        print()


if __name__ == "__main__":
    with open(
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\2048\config.json",
        "r",
    ) as f:
        config = json.load(f)

    ga = GA2048(save_dir=config["save_dir"])
    ga.learn(config)
