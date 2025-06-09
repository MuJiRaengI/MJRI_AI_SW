import os
import sys

sys.path.append(os.path.abspath("."))

import gym
import pygame

from source.envs.env import Env


class BreakoutEnv(Env):
    def __init__(self):
        super().__init__()

        pygame.init()
        pygame.display.set_caption("Breakout Controller (A: Left, D: Right)")

        self.env = gym.make("ALE/Breakout-v5", render_mode="human")
        self.obs, self.info = self.env.reset()
        self.done = False

    def _self_play(self):
        pass

    def _random_paly(self):
        pass

    def _train(self):
        pass

    def _test(self):
        pass
