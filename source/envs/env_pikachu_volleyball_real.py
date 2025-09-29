import numpy as np
import time
import gymnasium as gym
import gymnasium.spaces as spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
from source.utils.mjri_screen import MJRIScreen


class EnvPikachuVolleyBallReal(gym.Env):
    def __init__(self, screen_pos: Tuple[int, int, int, int], frame_stack=2, fps=60):
        self.screen_pos = screen_pos
        self.frame_stack = frame_stack
        self.fps = fps
        self.screen = MJRIScreen(
            x=screen_pos[0],
            y=screen_pos[1],
            w=screen_pos[2],
            h=screen_pos[3],
            buffer_size=self.frame_stack,
            thread_run=True,
            fps=self.fps,
        )

    def reset(self):
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        screen = self.screen.get_screen_buffer()
        return np.array(screen)
