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

from source.algorithm.genetic_algorithm.ga import GeneticAlgorithm
from source.algorithm.genetic_algorithm.network_reporter import NetworkReporter
from source.algorithm.genetic_algorithm.save_best_reporter import SaveBestReporter
from source.algorithm.genetic_algorithm.utils import *


class GALunarLander(GeneticAlgorithm):
    def __init__(
        self,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)

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
    with open(
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\lunarlander\config.json",
        "r",
    ) as f:
        config = json.load(f)

    ga = GALunarLander(save_dir=config["save_dir"])

    # ga.learn(config)

    winner_path = os.path.join(
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\results\GA\lunarlander\2025y09m06d_16h31m07s\best_genomes",
        "rank_01_fitness_-35.28_gen_4_id_611.pkl",
    )
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

    print()
