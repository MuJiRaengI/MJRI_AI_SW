import os
import sys

sys.path.append(os.path.abspath("."))
import time

import pygame
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)

from source.env_runner import EnvRunner
from source.env_runner.utils import *


class EnvPikachuVolleyBallRealRunner(EnvRunner):
    def __init__(self):
        super().__init__()
        self.env_id = "EnvPikachuVolleyBallReal"
        self.scale = 1

    def load_agent(self, agent_type: str):
        if agent_type.lower() == "PPO".lower():
            from source.core.reinforcement_learning.ppo import PPOPikachuVolleyBallReal

            return PPOPikachuVolleyBallReal(self.load_json(self.config_path))

        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def get_best_model_path(self, config: dict, result_path: str = None):
        agent_type = config.get("agent_type", None)
        if agent_type.lower() == "PPO".lower():
            ckpt_dir_name = "best_models"
            best_model_path = None
            latest_result_dir = result_path

            if os.path.exists(latest_result_dir):
                ckpt_path_list = sort_checkpoint_files_by_reward(
                    os.path.join(latest_result_dir, ckpt_dir_name), ext=".zip"
                )
                if len(ckpt_path_list) == 0:
                    print(f"No checkpoint files found in {latest_result_dir}.")
                    return None
                best_model_path = os.path.join(
                    latest_result_dir, ckpt_dir_name, ckpt_path_list[0]
                )
            else:
                print(f"No winner genome found in {latest_result_dir}.")
                return None
            return best_model_path

        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def set_test_mode(self, config: dict, agent, result_path: str = None):
        best_path = self.get_best_model_path(config, result_path)
        agent.prepare_predict(best_path)
        agent.env_num = 1
        return agent

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽, SPACE: FIRE(시작/재시작)\n"

    def _self_play(self, *args, **kwargs):
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self, *args, **kwargs):
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self, *args, **kwargs):
        try:
            config = self.load_json(self.config_path)
            if "screen_pos" in kwargs:
                config["screen_pos"] = kwargs["screen_pos"]
                self.save_json(self.config_path, config)

            agent_type = config["agent_type"]
            agent = self.load_agent(agent_type)
            agent.learn()

        finally:
            # 학습 완료 신호
            if self.training_queue is not None:
                self.training_queue.put(("done", None))

    def _test(self, *args, **kwargs):
        try:
            self.running = True
            config = self.load_json(self.config_path)

            result_dir = kwargs.get("result_dir", None)
            if result_dir is None:
                print(f"Result_dir is not exist.")
                if self.render_queue is not None:
                    self.render_queue.put(("done", None))
                return

            result_path = os.path.join(config["save_dir"], result_dir)

            agent_type = config["agent_type"]

            agent = self.load_agent(agent_type)
            best_model_path = self.get_best_model_path(config, result_path)
            agent = self.set_test_mode(config, agent, result_path)
            env = agent.make_env(mode="test", render_mode="human")

            obs, info = env.reset()
            step = 0
            while self.running:
                if self.render_queue is not None and not self.render_queue.empty():
                    msg = self.render_queue.get()
                    if isinstance(msg, tuple) and msg[0] == "stop":
                        break
                try:
                    new_best_model_path = self.get_best_model_path(config, result_path)
                    if new_best_model_path != best_model_path:
                        agent = self.set_test_mode(config, agent, result_path)
                        best_model_path = new_best_model_path
                        print(f"🔄 Loaded new best model from {best_model_path}")
                except Exception as e:
                    time.sleep(1)
                    print(f"Waiting for the model to be ready... {e}")

                if agent is None:
                    time.sleep(1)
                    continue

                action = agent.predict(obs, info["legal_actions"])
                done = False
                obs, reward, terminated, truncated, info = env.step(action)
                step += 1

                if terminated or truncated:
                    done = True

                print(env.render_obs(obs))
                print()
                if done:
                    time.sleep(1)
                    obs, info = env.reset()
                    step = 0
                    break

            env.close()
        finally:
            if self.render_queue is not None:
                self.render_queue.put(("done", None))


if __name__ == "__main__":
    runner = EnvPikachuVolleyBallRealRunner()
    config_path = r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\pikachu_volleyball_real\config_ppo_mlp.json"
    runner.play(
        config_path=config_path,
        mode="train",
        queue=None,
        screen_pos=(35, 373, 1215, 860),
        result_dir=r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\sol_pikachu_volleyball_real\results\2025y09m25d_23h55m08s",
    )
