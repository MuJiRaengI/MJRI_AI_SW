import os
import sys

sys.path.append(os.path.abspath("."))

import json
import time
import neat
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import numpy as np
import gymnasium as gym

from source.core.genetic_algorithm import GeneticAlgorithm
from source.core.genetic_algorithm.reporter import *
from source.core.genetic_algorithm.reporter.utils import *


class GALunarLander(GeneticAlgorithm):
    def __init__(self, config: dict):
        super().__init__(config)

    def make_env(self, render_mode=None):
        return gym.make("LunarLander-v3", render_mode=render_mode)

    def get_reporters(self, config):
        node_names = {
            -1: "X Position",
            -2: "Y Position",
            -3: "X Velocity",
            -4: "Y Velocity",
            -5: "Angle",
            -6: "Angular Velocity",
            -7: "Left Leg Contact",
            -8: "Right Leg Contact",
            0: "No Action",
            1: "Left Engine",
            2: "Main Engine",
            3: "Right Engine",
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

    # --------------------------
    def eval_genomes(self, genomes, config):
        for genome_id, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, config)
            env = self.make_env()

            fitness = 0
            for episode in range(self.num_episodes):
                observation, _ = env.reset()
                episode_reward = 0

                for step in range(self.max_step):
                    output = net.activate(observation)
                    action = output.index(max(output))
                    observation, reward, terminated, truncated, _ = env.step(action)
                    episode_reward += reward

                    if terminated or truncated:
                        break
                fitness += episode_reward

            env.close()
            genome.fitness = fitness / self.num_episodes


if __name__ == "__main__":
    pass
