import torch.nn as nn
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor

from source.core.reinforcement_learning import ReinforcementLearning
from source.envs import EnvPikachuVolleyBallReal
from source.core.reinforcement_learning.ppo.callback import (
    MonitorAndTimedCheckpointCallback,
    EnhancedMonitorAndCheckpointCallback,
)


class PPOPikachuVolleyBallReal(ReinforcementLearning):
    def __init__(self, config: dict):
        super().__init__(config)
        self.env_num = 1

        self.env = self.make_env()
        self.agent = None

        self.policy = config["policy"]
        self.activation_fn = config["activation_fn"]
        if self.activation_fn.lower() == "relu":
            self.activation_fn = nn.ReLU
        else:
            raise NotImplementedError(
                f"지원하지 않는 activation_fn 입니다: {self.policy_kwargs['activation_fn']}"
            )
        self.policy_kwargs = {
            "activation_fn": self.activation_fn,
            "net_arch": {
                "pi": config["net_arch_pi"],
                "vf": config["net_arch_vf"],
            },
        }
        self.n_steps = config["n_steps"]
        self.batch_size = config["batch_size"]
        self.n_epochs = config["n_epochs"]
        self.lr = config["lr"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self.clip_range = config["clip_range"]
        self.vf_coef = config["vf_coef"]
        self.ent_coef = config["ent_coef"]
        self.max_grad_norm = config["max_grad_norm"]
        self.device = config["device"]

        self.total_timesteps = config["total_timesteps"]
        self.callback = None

    def make_env(self, *args, **kwargs):
        env = EnvPikachuVolleyBallReal(screen_pos=self.config["screen_pos"])
        return env

    def load_callback(self):
        return EnhancedMonitorAndCheckpointCallback(
            self.logger,
            rolling_n=512,
            save_interval_sec=300,
            save_dir=self.save_dir,
            save_sub_dir="best_models",
            save_basename="latest_model",
            save_on_train_end=True,
            verbose=1,
        )

    def learn(self):
        self.create_dir()

        self.agent = MaskablePPO(
            policy=self.policy,
            env=self.env,
            policy_kwargs=self.policy_kwargs,
            n_steps=self.n_steps,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            learning_rate=self.lr,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            clip_range=self.clip_range,
            vf_coef=self.vf_coef,
            ent_coef=self.ent_coef,
            max_grad_norm=self.max_grad_norm,
            verbose=1,
            device=self.device,
        )
        self.callback = self.load_callback()
        self.agent.learn(
            total_timesteps=self.total_timesteps,
            callback=self.callback,
        )

    def prepare_predict(self, best_path):
        self.agent = MaskablePPO.load(best_path)

    def predict(self, obs, action_mask):
        action, _states = self.agent.predict(
            obs, action_masks=action_mask, deterministic=True
        )
        return action
