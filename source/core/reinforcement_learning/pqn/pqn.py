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


class PQN(ReinforcementLearning):
    def __init__(self, config: dict):
        super().__init__(config)

    def make_env(
        self,
        env_id: str,
        max_episode_steps: int = None,
        render_mode: str = None,
        frame_stack: int = 4,
        frame_skip: int = 1,
        noop_max: int = 30,
        screen_size: tuple = (84, 84),
        terminal_on_life_loss: bool = True,
        grayscale_obs: bool = True,
        scale_obs: bool = False,
        seed: int = None,
    ):
        def _init():
            # 멀티프로세싱 환경에서 각 워커에서 ale_py 등록
            import gymnasium as gym
            import ale_py

            gym.register_envs(ale_py)

            env = gym.make(
                env_id,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                frameskip=1,
            )
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=noop_max,
                frame_skip=frame_skip,
                screen_size=screen_size,
                terminal_on_life_loss=terminal_on_life_loss,
                grayscale_obs=grayscale_obs,
                scale_obs=scale_obs,
            )
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = gym.wrappers.FrameStackObservation(env, frame_stack)

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
        env_id: str,
        max_episode_steps: int = None,
        render_mode: str = None,
        frame_stack: int = 4,
        frame_skip: int = 1,
        noop_max: int = 30,
        screen_size: tuple = (84, 84),
        terminal_on_life_loss: bool = True,
        grayscale_obs: bool = True,
        scale_obs: bool = False,
        seed: int = None,
    ):
        env_fns = [
            self.make_env(
                env_id,
                max_episode_steps=max_episode_steps,
                render_mode=render_mode,
                frame_stack=frame_stack,
                frame_skip=frame_skip,
                noop_max=noop_max,
                screen_size=screen_size,
                terminal_on_life_loss=terminal_on_life_loss,
                grayscale_obs=grayscale_obs,
                scale_obs=scale_obs,
                seed=seed + i if seed is not None else None,
            )
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
        env.name = env_id
        return env

    def preprocess(self, obs):
        # obs: (B, H, W, C) or (H, W, C)
        if isinstance(obs, np.ndarray):
            if len(obs.shape) == 3:
                obs = obs[np.newaxis, ...]  # Add batch dimension
            obs = torch.from_numpy(obs).float()  # (B, H, W, C)
        return obs / 255.0  # Normalize to [0, 1]

    def load_model(self, model_type: str):
        if model_type.lower() == "pqn_cnn":
            from source.core.reinforcement_learning.model.pqn import QNetwork

            model = QNetwork(
                action_dim=self.action_dim,
                in_channels=self.frame_stack,
                norm_type="layer_norm",
                norm_input=False,
                input_size=self.screen_size,
                channels=self.channels,
            )
        else:
            raise ValueError(f"지원하지 않는 알고리즘 타입입니다: {model_type}")
        return model

    def eps_greedy_exploration(self, q_vals, eps):
        batch_size = q_vals.shape[0]
        greedy_actions = torch.argmax(q_vals, dim=-1)
        random_actions = torch.randint(
            0, q_vals.shape[-1], (batch_size,), device=self.device
        )
        random_mask = torch.rand(batch_size, device=self.device) < eps
        chosen_actions = torch.where(random_mask, random_actions, greedy_actions)
        return chosen_actions

    def eps_scheduler(
        self,
        step: int,
        eps_start: float,
        eps_finish: float,
        eps_decay: float,
        num_updates_decay: int,
    ):
        decay_steps = eps_decay * num_updates_decay
        progress = min(step / decay_steps, 1.0)
        return eps_start + progress * (eps_finish - eps_start)

    def lr_scheduler(
        self,
        step: int,
        lr: float,
        lr_linear_decay: bool,
        num_minibatches: int,
        num_epochs: int,
        num_updates_decay: int,
    ):
        if lr_linear_decay:
            total_steps = num_updates_decay * num_minibatches * num_epochs
            progress = min(step / total_steps, 1.0)
            return lr + progress * (1e-20 - lr)
        return lr

    @torch.no_grad()
    def compute_lambda_targets_fixed(self, last_q, q_vals, rewards, dones, gamma, lam):
        # q_vals: (T, B, A), rewards/dones: (T, B), last_q: (B,)
        T, B, A = q_vals.shape
        targets = torch.empty((T, B), device=last_q.device, dtype=torch.float32)

        # t = T-1
        lambda_returns = rewards[-1] + gamma * (1.0 - dones[-1]) * last_q
        targets[-1] = lambda_returns

        for t in reversed(range(T - 1)):
            next_q = q_vals[t + 1].max(dim=-1).values  # Q(s_{t+1},·)
            target_bootstrap = rewards[t] + gamma * (1.0 - dones[t]) * next_q
            delta = lambda_returns - next_q
            lambda_returns = target_bootstrap + gamma * lam * delta
            lambda_returns = (1.0 - dones[t]) * lambda_returns + dones[t] * rewards[t]
            targets[t] = lambda_returns

        return targets

    def _train(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.model.to(self.device)
        self.optimizer = torch.optim.RAdam(self.model.parameters(), lr=self.lr)
        train_state = CustomTrainState(
            model=self.model,
            optimizer=self.optimizer,
        )

        current_obs = self.env.reset()
        info = None
        if isinstance(current_obs, tuple):
            current_obs, info = current_obs

        # Ensure proper shape for vectorized envs
        if len(current_obs.shape) == 3:  # Single env
            current_obs = current_obs[np.newaxis, ...]  # Add batch dimension

        # Training loop
        all_metrics = []

        for update in tqdm(range(self.num_updates)):
            train_state, obs, metrics = self._update_step(current_obs, train_state)
            all_metrics.append(metrics)
            current_obs = obs

            # 설정된 간격으로 best 모델 저장 체크
            if (update + 1) % self.save_best_interval == 0:
                saved = self.check_and_save_best(self.model, step=train_state.timesteps)
                if saved:
                    mean_reward = self.get_mean_reward()
                    self.logger.info(
                        f"Update {update + 1}: New best model saved! Mean reward: {mean_reward:.3f}"
                    )

        # _train() 메서드가 metrics를 반환하도록 수정
        return {"metrics": all_metrics}

    def _update_step(self, current_obs, train_state):
        # SAMPLE PHASE
        transitions_list = []
        episode_returns = []
        episode_lengths = []
        # Keep track of episode rewards and lengths manually - separate train and test
        episode_rewards = np.zeros(self.total_envs)
        episode_steps = np.zeros(self.total_envs, dtype=int)
        test_episode_returns = []
        test_episode_lengths = []
        obs = current_obs.copy()
        for step in range(self.num_steps):
            # Convert numpy to torch tensor and move to device
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            # Set network to eval mode for action selection
            self.model.eval()
            with torch.no_grad():
                q_vals = self.model(obs_tensor)

            # Epsilon for each environment
            eps_values = np.full(
                self.total_envs,
                self.eps_scheduler(
                    step=train_state.n_updates,
                    eps_start=self.eps_start,
                    eps_finish=self.eps_finish,
                    eps_decay=self.eps_decay,
                    num_updates_decay=self.num_updates_decay,
                ),
            )
            if self.test_during_training:
                eps_values[-self.test_env_num :] = 0.0  # Greedy for test envs

            # Choose actions
            actions = self.eps_greedy_exploration(
                q_vals, torch.from_numpy(eps_values).to(self.device)
            )
            actions_np = actions.cpu().numpy()

            # Step environment
            step_result = self.env.step(actions_np)
            if len(step_result) == 4:  # (obs, reward, done, info)
                next_obs, reward, done, info = step_result
                truncated = np.zeros_like(done)
            else:  # (obs, reward, terminated, truncated, info)
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated | truncated

            # Update episode tracking
            episode_rewards += reward
            episode_steps += 1

            # Check for episode completion and collect stats - separate train and test
            for env_idx in range(self.total_envs):
                if done[env_idx]:
                    if self.test_during_training and env_idx >= self.env_num:
                        # This is a test environment
                        test_episode_returns.append(episode_rewards[env_idx])
                        test_episode_lengths.append(episode_steps[env_idx])
                    else:
                        # This is a training environment
                        episode_returns.append(episode_rewards[env_idx])
                        episode_lengths.append(episode_steps[env_idx])

                        # Agent 클래스에 에피소드 보상 기록
                        self.update_episode_rewards(episode_rewards[env_idx])

                        # 충분한 에피소드가 누적되면 best 모델 저장 체크
                        if len(self.episode_rewards) >= 10:  # 최소 10개 에피소드 필요
                            saved = self.check_and_save_best(
                                self.model, step=train_state.timesteps
                            )
                            if saved:
                                mean_reward = self.get_mean_reward()
                                self.logger.info(
                                    f"Episode completed: New best model saved! Mean reward: {mean_reward:.3f}"
                                )

                    episode_rewards[env_idx] = 0
                    episode_steps[env_idx] = 0

            transition = Transition(
                obs=torch.from_numpy(obs).float(),
                action=actions.cpu(),
                reward=torch.from_numpy(reward * self.rew_scale).float(),
                done=torch.from_numpy(done.astype(np.float32)),
                next_obs=torch.from_numpy(next_obs).float(),
                q_val=q_vals.cpu(),
            )
            transitions_list.append(transition)
            obs = next_obs

        # Stack transitions
        transitions = Transition(
            obs=torch.stack([t.obs for t in transitions_list]),
            action=torch.stack([t.action for t in transitions_list]),
            reward=torch.stack([t.reward for t in transitions_list]),
            done=torch.stack([t.done for t in transitions_list]),
            next_obs=torch.stack([t.next_obs for t in transitions_list]),
            q_val=torch.stack([t.q_val for t in transitions_list]),
        )

        # Remove testing envs from training data
        if self.test_during_training:
            transitions = Transition(
                obs=transitions.obs[:, : -self.test_env_num],
                action=transitions.action[:, : -self.test_env_num],
                reward=transitions.reward[:, : -self.test_env_num],
                done=transitions.done[:, : -self.test_env_num],
                next_obs=transitions.next_obs[:, : -self.test_env_num],
                q_val=transitions.q_val[:, : -self.test_env_num],
            )

        train_state.timesteps += self.num_steps * self.env_num

        # Compute targets using lambda returns
        self.model.eval()  # Set to eval mode for target computation
        with torch.no_grad():
            last_obs_tensor = transitions.next_obs[-1].to(self.device)
            last_q = self.model(last_obs_tensor).max(-1).values
            lambda_targets = self.compute_lambda_targets_fixed(
                last_q=last_q,
                q_vals=transitions.q_val.to(self.device),
                rewards=transitions.reward.to(self.device),
                dones=transitions.done.to(self.device),
                gamma=self.gamma,
                lam=self.lam,
            )

        # NETWORKS UPDATE
        total_loss = 0
        total_qvals = 0

        # Set network to train mode for parameter updates
        self.model.train()

        for epoch in range(self.num_epochs):
            # Prepare minibatches
            batch_size = self.num_steps * self.env_num
            minibatch_size = batch_size // self.num_minibatches

            # Flatten and shuffle
            obs_flat = transitions.obs.reshape(-1, *transitions.obs.shape[2:])
            action_flat = transitions.action.reshape(-1)
            targets_flat = lambda_targets.reshape(-1)

            indices = torch.randperm(batch_size)

            for i in range(self.num_minibatches):
                start_idx = i * minibatch_size
                end_idx = start_idx + minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                obs_batch = obs_flat[minibatch_indices].to(self.device)
                action_batch = action_flat[minibatch_indices].to(self.device)
                target_batch = targets_flat[minibatch_indices].to(self.device)

                # Update learning rate if linear decay is enabled
                if self.lr_linear_decay:
                    cur_lr = self.lr_scheduler(
                        step=train_state.grad_steps,
                        lr=self.lr,
                        lr_linear_decay=self.lr_linear_decay,
                        num_minibatches=self.num_minibatches,
                        num_epochs=self.num_epochs,
                        num_updates_decay=self.num_updates_decay,
                    )
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = cur_lr

                # Forward pass
                q_vals = self.model(obs_batch)
                chosen_action_qvals = q_vals.gather(
                    1, action_batch.unsqueeze(1)
                ).squeeze(1)

                # Compute loss
                loss = 0.5 * F.mse_loss(chosen_action_qvals, target_batch)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()
                total_qvals += chosen_action_qvals.mean().item()
                train_state.grad_steps += 1

        train_state.n_updates += 1

        # Prepare metrics
        metrics = {
            "env_step": train_state.timesteps,
            "update_steps": train_state.n_updates,
            "env_frame": train_state.timesteps * self.frame_stack,
            "grad_steps": train_state.grad_steps,
            "td_loss": total_loss / (self.num_epochs * self.num_minibatches),
            "qvals": total_qvals / (self.num_epochs * self.num_minibatches),
        }

        # Add episode metrics - separate train and test
        if episode_returns:
            metrics["episode_return"] = np.mean(episode_returns)
            metrics["episode_length"] = np.mean(episode_lengths)

        # Add test metrics if available
        if self.test_during_training and test_episode_returns:
            metrics["test/episode_return"] = np.mean(test_episode_returns)
            metrics["test/episode_length"] = np.mean(test_episode_lengths)

        return train_state, obs, metrics

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
        self.max_step = self.config["max_step"]
        self.render_mode = self.config["render_mode"]
        self.frame_stack = self.config["frame_stack"]
        self.frame_skip = self.config["frame_skip"]
        self.noop_max = self.config["noop_max"]
        self.screen_size = tuple(self.config["screen_size"])
        self.terminal_on_life_loss = self.config["terminal_on_life_loss"]
        self.grayscale_obs = self.config["grayscale_obs"]
        self.scale_obs = self.config["scale_obs"]
        self.seed = self.config.get("seed", None)

        # model config
        self.device = self.config["device"]
        self.model_type = self.config["model_type"]
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
            env_id=self.env_id,
            max_episode_steps=self.max_step,
            render_mode=self.render_mode,
            frame_stack=self.frame_stack,
            frame_skip=self.frame_skip,
            noop_max=self.noop_max,
            screen_size=self.screen_size,
            terminal_on_life_loss=self.terminal_on_life_loss,
            grayscale_obs=self.grayscale_obs,
            scale_obs=self.scale_obs,
            seed=self.seed,
        )

        self.model = self.load_model(self.model_type)
        if self.pretrained_path is not None and os.path.exists(self.pretrained_path):
            self.logger.info(f"Pre-trained 모델 로드: {self.pretrained_path}")
            self.model.load_state_dict(
                torch.load(self.pretrained_path, map_location=self.device)
            )

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

    def prepare_predict(self):
        # env config
        self.env_id = self.config["env_id"]
        self.action_dim = self.config["action_dim"]
        self.max_step = self.config["max_step"]
        self.frame_stack = self.config["frame_stack"]
        self.frame_skip = self.config["frame_skip"]
        self.noop_max = self.config["noop_max"]
        self.screen_size = tuple(self.config["screen_size"])
        self.grayscale_obs = self.config["grayscale_obs"]
        self.scale_obs = self.config["scale_obs"]
        self.seed = self.config.get("seed", None)

        # model config
        self.device = self.config["device"]
        self.model_type = self.config["model_type"]
        self.channels = self.config["channels"]

        self.model = self.load_model(self.model_type)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, obs):
        if self.model is None:
            raise ValueError("Model is not loaded. Call prepare_predict() first.")

        output = self.model(self.preprocess(obs).to(self.device))
        action = output.argmax(dim=-1).cpu().numpy()
        return action.item()
