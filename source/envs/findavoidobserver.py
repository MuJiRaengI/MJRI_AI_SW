import os
import pygame
import numpy as np
import PySide6.QtWidgets as QtWidgets
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

    def _show_manual_play_message(self):
        try:

            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle("키보드 조작 안내")
            msg.setText(
                "[조작법] 방향키: D(→), C(↘), S(↓), Z(↙), A(←), Q(↖), W(↑), E(↗)\nESC: 종료"
            )
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.exec()
        except Exception:
            pass

    def _self_play(self):
        self._show_manual_play_message()
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
        pass

    def _test(self, solution_dir):
        pass
