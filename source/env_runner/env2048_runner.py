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


class Env2048Runner(EnvRunner):
    def __init__(self):
        super().__init__()
        self.env_id = "Env2048"
        self.scale = 1

    def load_agent(self, agent_type: str):
        if agent_type.lower() == "PQN".lower():
            from source.core.reinforcement_learning.pqn import PQNEnv2048

            return PQNEnv2048(self.load_json(self.config_path))

        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {agent_type}")

    def get_best_model_path(self, config: dict, result_path: str = None):
        agent_type = config.get("agent_type", None)
        if agent_type == "PQN":
            best_model_name = "model_final_0.pth"
            ckpt_dir_name = "best_models"

            best_model_path = None
            if result_path is None:
                # find winner path
                result_dirs = sorted(os.listdir(config["save_dir"]))
                if len(result_dirs) == 0:
                    print(f"No result directories found in {config['save_dir']}.")
                    return None

                latest_result_dir = os.path.join(config["save_dir"], result_dirs[-1])
            else:
                latest_result_dir = result_path

            best_genome_path = os.path.join(latest_result_dir, best_model_name)
            if os.path.exists(best_genome_path):
                best_model_path = best_genome_path
            elif os.path.exists(latest_result_dir):
                ckpt_path_list = sort_checkpoint_files_by_reward(
                    os.path.join(latest_result_dir, ckpt_dir_name)
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
        agent.prepare_predict()
        return agent

    def key_info(self) -> str:
        return "[조작법] A: 왼쪽, D: 오른쪽, SPACE: FIRE(시작/재시작)\n"

    def _self_play(self, *args, **kwargs):
        env = gym.make(self.env_id, render_mode="rgb_array")
        obs, info = env.reset()

        pygame.init()
        frame = env.render()
        h, w, _ = frame.shape
        h, w = h * self.scale, w * self.scale
        screen = pygame.display.set_mode((h, w))
        pygame.display.set_caption(
            "Breakout Manual Play (A: Left, D: Right, SPACE: Fire)"
        )
        clock = pygame.time.Clock()

        running = True
        action = 0

        while running:
            # 윈도우 종료 처리
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return

            # render_queue로부터 stop 신호를 받으면 중단
            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 키보드 입력 처리 (pygame 방식)
            keys = pygame.key.get_pressed()
            if keys[pygame.K_a]:
                action = 3  # LEFT
            elif keys[pygame.K_d]:
                action = 2  # RIGHT
            elif keys[pygame.K_SPACE]:
                action = 1  # FIRE
            else:
                action = 0  # NOOP

            # 환경 업데이트
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()

            # 화면 출력
            # surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            # screen.blit(surface, (0, 0))
            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            # 에피소드 끝났으면 리셋
            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _random_play(self, *args, **kwargs):
        env = gym.make(self.env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = self.fps
        obs, info = env.reset()

        pygame.init()
        frame = env.render()
        h, w, _ = frame.shape
        h, w = h * self.scale, w * self.scale
        screen = pygame.display.set_mode((h, w))
        pygame.display.set_caption("Breakout Random Play (Pygame UI)")
        clock = pygame.time.Clock()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    pygame.quit()
                    if self.render_queue is not None:
                        self.render_queue.put(("done", None))
                    return

            if self.render_queue is not None and not self.render_queue.empty():
                msg = self.render_queue.get()
                if isinstance(msg, tuple) and msg[0] == "stop":
                    break

            # 랜덤 액션
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            frame = env.render()

            scaled_frame = pygame.transform.scale(
                pygame.surfarray.make_surface(frame.swapaxes(0, 1)),
                (h, w),
            )
            screen.blit(scaled_frame, (0, 0))
            pygame.display.flip()

            if terminated or truncated:
                obs, info = env.reset()

            clock.tick(self.fps)

        env.close()
        pygame.quit()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))

    def _train(self, *args, **kwargs):
        try:
            config = self.load_json(self.config_path)

            agent_type = config["agent_type"]
            agent = self.load_agent(agent_type)
            agent.learn()

        finally:
            # 학습 완료 신호
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

        result_path = os.path.join(config["save_dir"], result_dir)

        env_id = config["env_id"]
        max_step = None
        render_mode = "human"
        frame_stack = config["frame_stack"]
        frame_skip = config["frame_stack"]
        noop_max = config["noop_max"]
        screen_size = tuple(config["screen_size"])
        terminal_on_life_loss = config["terminal_on_life_loss"]
        grayscale_obs = config["grayscale_obs"]
        scale_obs = config["scale_obs"]

        agent_type = config["agent_type"]

        agent = self.load_agent(agent_type)
        best_model_path = self.get_best_model_path(config, result_path)
        agent = self.set_test_mode(config, agent, result_path)
        env = agent.make_env(
            env_id,
            max_episode_steps=max_step,
            render_mode=render_mode,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            noop_max=noop_max,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            scale_obs=scale_obs,
        )()

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

            action = agent.predict(obs)
            done = False
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1

            if terminated or truncated:
                done = True

            if max_step is not None and step >= max_step:
                done = True

            if done:
                obs, _ = env.reset()
                step = 0

        env.close()
        if self.render_queue is not None:
            self.render_queue.put(("done", None))


if __name__ == "__main__":
    runner = Env2048Runner()
    config_path = (
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\env2048\config_pqn_mlp.json"
    )
    runner.play(
        config_path=config_path,
        mode="train",
        queue=None,
        result_dir=r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\sol_env2048_pqn_mlp\results\2025y09m21d_19h03m12s",
    )
