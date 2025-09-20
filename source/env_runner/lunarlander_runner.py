import os
import sys

sys.path.append(os.path.abspath("."))

import re
import time
import neat
from datetime import datetime

# import gym
import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
from source.env_runner import EnvRunner

from source.core import Agent
from source.env_runner.utils import *


class LunarLanderRunner(EnvRunner):
    def __init__(self):
        super().__init__()

    def load_agent(self, agent_type: str):
        if agent_type == "GA":
            from source.core.genetic_algorithm import GALunarLander

            return GALunarLander(self.load_json(self.config_path))
        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def get_best_model_path(self, config: dict):
        agent_type = config.get("type", None)
        if agent_type == "GA":
            best_model_name = "best_genome.pkl"
            ckpt_dir_name = "best_genomes"

            winner_path = None
            # find winner path
            result_dirs = sorted(os.listdir(config["save_dir"]))
            if len(result_dirs) == 0:
                print(f"No result directories found in {config['save_dir']}.")
                return None

            latest_result_dir = os.path.join(config["save_dir"], result_dirs[-1])

            best_genome_path = os.path.join(latest_result_dir, best_model_name)
            if os.path.exists(best_genome_path):
                winner_path = best_genome_path
            elif os.path.exists(latest_result_dir):
                ckpt_path_list = sort_checkpoint_files_by_fitness(
                    os.path.join(latest_result_dir, ckpt_dir_name)
                )
                if len(ckpt_path_list) == 0:
                    print(f"No checkpoint files found in {latest_result_dir}.")
                    return None
                winner_path = os.path.join(
                    latest_result_dir, ckpt_dir_name, ckpt_path_list[0]
                )
            else:
                print(f"No winner genome found in {latest_result_dir}.")
                return None
            return winner_path

        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def set_test_mode(self, config: dict, agent: Agent):
        agent_type = config.get("type", None)
        if agent_type == "GA":
            from source.core.genetic_algorithm import GALunarLander

            if not isinstance(agent, GALunarLander):
                print("Agent is not an instance of GALunarLander.")
                return None

            winner_path = self.get_best_model_path(config)

            ga_config_path = config["ga_config_path"]
            neat_config = neat.Config(
                neat.DefaultGenome,
                neat.DefaultReproduction,
                neat.DefaultSpeciesSet,
                neat.DefaultStagnation,
                ga_config_path,
            )

            agent.load_winner_net(winner_path, neat_config)

            return agent
        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def key_info(self) -> str:
        return "해당 게임은 지원하지 않습니다"

    def _self_play(self, *args, **kwargs):
        if self.render_queue is not None:
            self.render_queue.put(("done", None))
        return

    def _random_play(self, *args, **kwargs):
        if self.render_queue is not None:
            self.render_queue.put(("done", None))
        return

    def _train(self, *args, **kwargs):
        config = self.load_json(self.config_path)

        agent_type = config.get("type", None)
        agent = self.load_agent(agent_type)
        agent.learn()

        if self.training_queue is not None:
            self.training_queue.put(("done", None))

    def _test(self, *args, **kwargs):
        self.running = True
        config = self.load_json(self.config_path)

        result_dir = kwargs.get("result_dir", None)
        if result_dir is None:
            print(f"Result_dir is not exist.")
            if self.render_queue is not None:
                self.render_queue.put(("done", None))
            return

        agent_type = config.get("type", None)
        agent = self.load_agent(agent_type)
        best_model_path = self.get_best_model_path(config)
        agent = self.set_test_mode(config, agent)
        env = agent.make_env(render_mode="human")
        max_step = config["max_step"]

        obs, _ = env.reset()
        step = 0
        while self.running:
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            new_best_model_path = self.get_best_model_path(config)
            if new_best_model_path != best_model_path:
                try:
                    agent = self.set_test_mode(config, agent)
                    best_model_path = new_best_model_path
                    print(f"🔄 Loaded new best model from {best_model_path}")
                except Exception as e:
                    time.sleep(1)
                    print(f"Waiting for the model to be ready... {e}")

            if agent is None:
                time.sleep(1)
                continue

            action = agent.predict(obs)
            done = False
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                done = True

            if step >= max_step:
                done = True

            if done:
                obs, _ = env.reset()
                step = 0

        env.close()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    runner = LunarLanderRunner()
    config_path = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\lunarlander\config_ga.json"
    )
    # runner.play(config_path=config_path, mode="train")
    runner.play(
        config_path=config_path,
        mode="test",
        result_dir=r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\sol_lunarlander\results\2025y09m21d_00h07m53s",
    )
