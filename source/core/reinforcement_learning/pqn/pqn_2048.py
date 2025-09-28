import os
import time

import gymnasium as gym
import ale_py

gym.register_envs(ale_py)
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from stable_baselines3.common.atari_wrappers import FireResetEnv

from source.core.reinforcement_learning import ReinforcementLearning
from source.core.reinforcement_learning.pqn.state import CustomTrainState
from source.core.reinforcement_learning.pqn.transition import Transition
from source.envs import Env2048
from source.core.reinforcement_learning.pqn import PQN


class PQNEnv2048(PQN):
    def __init__(self, config: dict):
        super().__init__(config)
        self.log_reward_threshold = 0.2
        self.log_reward_step = 300

    def make_env(
        self,
        seed: int = None,
        *args,
        **kwargs,
    ):
        def _init():
            env = Env2048(size=4)

            # Set seeds if provided
            if seed is not None:
                env.action_space.seed(seed)
                env.observation_space.seed(seed)

            return env

        return _init

    def make_env_vector(
        self,
        num_envs: int,
        using_async: bool,
        seed: int = None,
        *args,
        **kwargs,
    ):
        env_fns = [
            self.make_env(seed=seed + i if seed is not None else None)
            for i in range(num_envs)
        ]
        if num_envs == 1:
            env = env_fns[0]()
        else:
            if using_async:
                env = AsyncVectorEnv(env_fns)
            else:
                env = SyncVectorEnv(env_fns)

        if hasattr(env, "single_action_space"):
            action_space = env.single_action_space
            observation_space = env.single_observation_space
        else:
            action_space = env.action_space
            observation_space = env.observation_space
            env.single_action_space = action_space
            env.single_observation_space = observation_space

        env.num_envs = num_envs
        env.name = "Env2048"
        return env

    def preprocess(self, obs):
        # obs: (B, H, W, C) or (H, W, C)
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float()  # (B, H, W, C)

        # Apply log2 transformation (add 1 to handle 0 values)
        obs = torch.log2(obs + 1)

        # Normalize by theoretical max (2^17 = 131072 -> log2(131072+1) ≈ 17)
        obs = obs / 17.0

        B = obs.shape[0]
        obs = obs.view(B, -1)  # Flatten to (B, 16)
        return obs

    def load_model(self, model_type: str):
        if model_type.lower() == "pqn_cnn":
            from source.core.reinforcement_learning.model.pqn_mlp import QNetwork

            model = QNetwork(
                input_dim=self.in_channels,
                action_dim=self.action_dim,
                norm_type="layer_norm",
                norm_input=False,
                hidden_dims=self.channels,
            )
        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {model_type}")
        return model

    def learn(self):
        self.create_dir()

        # env config
        self.env_id = self.config["env_id"]
        self.action_dim = self.config["action_dim"]
        self.env_num = self.config["env_num"]
        self.test_env_num = self.config["test_env_num"]
        self.test_during_training = True if self.test_env_num > 0 else False
        self.total_envs = self.env_num + self.test_env_num
        self.using_async = self.config["using_async"]
        self.seed = self.config.get("seed", None)

        # model config
        self.device = self.config["device"]
        self.model_type = self.config["model_type"]
        self.in_channels = self.config["in_channels"]
        self.channels = self.config["channels"]
        self.pretrained_path = self.config["pretrained_path"]

        # training config
        self.total_timesteps = self.config["total_timesteps"]
        self.total_timesteps_decay = self.config["total_timesteps_decay"]
        self.num_epochs = self.config["num_epochs"]
        self.num_minibatches = self.config["num_minibatches"]
        self.num_steps = self.config["num_steps"]
        self.eps_start = self.config["eps_start"]
        self.eps_finish = self.config["eps_finish"]
        self.eps_decay = self.config["eps_decay"]
        self.rew_scale = self.config["rew_scale"]
        self.gamma = self.config["gamma"]
        self.lam = self.config["lam"]
        self.lr = self.config["lr"]
        self.lr_linear_decay = self.config["lr_linear_decay"]
        self.max_grad_norm = self.config["max_grad_norm"]

        # not used
        self.frame_stack = 1

        self.num_updates = self.total_timesteps // self.num_steps // self.env_num
        self.num_updates_decay = (
            self.total_timesteps_decay // self.num_steps // self.env_num
        )

        assert (
            self.num_steps * self.env_num
        ) % self.num_minibatches == 0, (
            "num_minibatches must divide num_steps * num_envs"
        )

        self.env = self.make_env_vector(
            num_envs=self.total_envs,
            using_async=self.using_async,
            seed=self.seed,
        )

        self.model = self.load_model(self.model_type)
        if self.pretrained_path is not None and os.path.exists(self.pretrained_path):
            self.logger.info(f"Pre-trained 모델 로드: {self.pretrained_path}")
            self.model.load_state_dict(
                torch.load(self.pretrained_path, map_location=self.device)
            )

        # 학습 시작 전 설정 정보 로깅
        self.logger.info("=" * 60)
        self.logger.info("PQN 학습 시작 - 설정 정보")
        self.logger.info("=" * 60)
        self.logger.info(f"환경: {self.env_id}")
        self.logger.info(
            f"총 스텝: {self.total_timesteps:,} | 총 업데이트: {self.num_updates:,}"
        )
        self.logger.info(
            f"환경 개수: {self.total_envs} (train: {self.env_num}, test: {self.test_env_num})"
        )
        self.logger.info(
            f"Epsilon: {self.eps_start:.4f} → {self.eps_finish:.4f} (decay: {self.eps_decay:.3f})"
        )
        self.logger.info(
            f"학습률: {self.lr:.6f} (linear decay: {self.lr_linear_decay})"
        )
        self.logger.info(f"할인율(γ): {self.gamma} | Lambda: {self.lam}")
        self.logger.info(f"Best 모델 저장 간격: {self.save_best_interval} 업데이트")
        self.logger.info(f"디바이스: {self.device}")
        if self.pretrained_path:
            self.logger.info(f"사전 훈련 모델: 사용")
        self.logger.info("=" * 60)

        self.start_time = time.time()

        outs = self._train()

        end_time = time.time()
        total_time = max(end_time - self.start_time, 1e-6)

        if self.logger:
            self.logger.info(
                f"Training completed. "
                f"Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes), "
                f"Total Steps: {self.total_timesteps}, "
                f"Steps per Second: {self.total_timesteps / total_time:.2f}"
            )

        # Last detailed_logging_freq updates
        final_metrics = outs["metrics"][-self.detailed_logging_freq :]
        if final_metrics and self.logger:
            avg_final_loss = np.mean([m["td_loss"] for m in final_metrics])
            avg_final_qvals = np.mean([m["qvals"] for m in final_metrics])

            episode_returns = [
                m.get("episode_return", 0)
                for m in final_metrics
                if "episode_return" in m
            ]

            if episode_returns:
                avg_final_return = np.mean(episode_returns)
                self.logger.info(
                    f"Final {self.detailed_logging_freq} Updates - "
                    f"Avg TD Loss: {avg_final_loss:.4f}, "
                    f"Avg QVals: {avg_final_qvals:.4f}, "
                    f"Avg Episode Return: {avg_final_return:.2f}"
                )
            else:
                self.logger.info(
                    f"Final {self.detailed_logging_freq} Updates - "
                    f"Avg TD Loss: {avg_final_loss:.4f}, "
                    f"Avg QVals: {avg_final_qvals:.4f}, "
                    f"No Episode Returns Recorded"
                )

        total_steps = self.total_timesteps
        steps_per_second = total_steps / total_time
        if self.logger:
            self.logger.info(
                f"Total Steps: {total_steps}, "
                f"Steps per Second: {steps_per_second:.2f}"
            )

        # Agent 클래스의 최종 모델 저장 기능 사용
        self.save_final_model(self.model, step=self.total_steps)

        self.model.cpu()
