import random
import os
import sys

sys.path.append(os.path.abspath("."))

from source.ai.rl.agent.agent import Agent


class BreakoutAgent(Agent):
    def __init__(self):
        super().__init__()

        # Breakout: 0=NOOP, 1=FIRE, 2=RIGHT, 3=LEFT
        self.noop = 0
        self.fire = 1
        self.right = 2
        self.left = 3
        self.done = False
        self.fps = 30

    def select_action(self, state):
        # Breakout: 0=NOOP, 2=RIGHT, 3=LEFT
        # 여기서는 FIRE(1)는 제외하고, 0/2/3 중 무작위 선택

        return random.choice([self.noop, self.fire, self.right, self.left])

    def step(self, state, action, reward, next_state, done):
        # 샘플: 학습 없음
        pass

    def learn(self, state, action, reward, next_state, done):
        # 샘플: 학습 없음
        pass
