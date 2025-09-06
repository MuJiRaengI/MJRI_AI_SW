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
from source.envs.env2048 import Env2048PG


class GA2048(GeneticAlgorithm):
    def __init__(
        self,
        size: int,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)
        self.size = size

    # --------------------------
    def eval_genomes(self, genomes, config):
        print()

    def test_winner(self, genome, config, episodes: int, steps: int):
        print()

    def get_reporters(self):
        return []


if __name__ == "__main__":
    with open(
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\lunarlander\config.json",
        "r",
    ) as f:
        config = json.load(f)

    ga = GA2048(size=4, save_dir=config["save_dir"])
    ga.learn(config)
