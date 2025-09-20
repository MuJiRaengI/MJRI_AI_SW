import os
import time
import json
from typing import TypeVar
from dataclasses import dataclass

ObsType = TypeVar("ObsType")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from stable_baselines3.common.atari_wrappers import NoopResetEnv, FireResetEnv

from source.ai.rl.agent import Agent


@dataclass
class Transition:
    obs: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    next_obs: torch.Tensor
    q_val: torch.Tensor


class CustomTrainState:
    def __init__(self, model, optimizer, timesteps=0, n_updates=0, grad_steps=0):
        self.model = model
        self.optimizer = optimizer
        self.timesteps = timesteps
        self.n_updates = n_updates
        self.grad_steps = grad_steps


class PQN(Agent):
    def __init__(
        self,
        model: nn.Module,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)
        self.model = model

        # architecture_info = model.body.get_architecture_info()
        # self.logger.info(f" QNetwork-body 모델 구조: {architecture_info}")

    def make_env(
        self,
        env_id: str,
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
            env = gym.make(env_id, frameskip=1, render_mode=render_mode)
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

    def _update_step(
        self,
        env: gym.Env,
        model: nn.Module,
        optimizer: optim.Optimizer,
        num_epochs: int,
        num_minibatches: int,
        frame_stack: int,
        train_state: CustomTrainState,
        current_obs: ObsType,
        total_envs: int,
        num_envs: int,
        num_test_envs: int,
        num_steps: int,
        test_during_training: bool,
        eps_start: float,
        eps_finish: float,
        eps_decay: float,
        num_updates_decay: int,
        rew_scale: float,
        gamma: float,
        lam: float,
        lr: float,
        lr_linear_decay: bool,
        max_grad_norm: float,
    ):

        # SAMPLE PHASE
        transitions_list = []
        episode_returns = []
        episode_lengths = []
        # Keep track of episode rewards and lengths manually - separate train and test
        episode_rewards = np.zeros(total_envs)
        episode_steps = np.zeros(total_envs, dtype=int)
        test_episode_returns = []
        test_episode_lengths = []

        obs = current_obs.copy()
        for step in range(num_steps):
            # Convert numpy to torch tensor and move to device
            obs_tensor = torch.from_numpy(obs).float().to(self.device)

            # Set network to eval mode for action selection
            model.eval()
            with torch.no_grad():
                q_vals = model(obs_tensor)

            # Epsilon for each environment
            eps_values = np.full(
                total_envs,
                self.eps_scheduler(
                    step=train_state.n_updates,
                    eps_start=eps_start,
                    eps_finish=eps_finish,
                    eps_decay=eps_decay,
                    num_updates_decay=num_updates_decay,
                ),
            )
            if test_during_training:
                eps_values[-num_test_envs:] = 0.0  # Greedy for test envs

            # Choose actions
            actions = self.eps_greedy_exploration(
                q_vals, torch.from_numpy(eps_values).to(self.device)
            )
            actions_np = actions.cpu().numpy()

            # Step environment
            step_result = env.step(actions_np)
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
            for env_idx in range(total_envs):
                if done[env_idx]:
                    if test_during_training and env_idx >= num_envs:
                        # This is a test environment
                        test_episode_returns.append(episode_rewards[env_idx])
                        test_episode_lengths.append(episode_steps[env_idx])
                    else:
                        # This is a training environment
                        episode_returns.append(episode_rewards[env_idx])
                        episode_lengths.append(episode_steps[env_idx])

                        # Agent 클래스에 에피소드 보상 기록
                        self.update_episode_rewards(episode_rewards[env_idx])

                    episode_rewards[env_idx] = 0
                    episode_steps[env_idx] = 0

            transition = Transition(
                obs=torch.from_numpy(obs).float(),
                action=actions.cpu(),
                reward=torch.from_numpy(reward * rew_scale).float(),
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
        if test_during_training:
            transitions = Transition(
                obs=transitions.obs[:, :-num_test_envs],
                action=transitions.action[:, :-num_test_envs],
                reward=transitions.reward[:, :-num_test_envs],
                done=transitions.done[:, :-num_test_envs],
                next_obs=transitions.next_obs[:, :-num_test_envs],
                q_val=transitions.q_val[:, :-num_test_envs],
            )

        train_state.timesteps += num_steps * num_envs

        # Compute targets using lambda returns
        model.eval()  # Set to eval mode for target computation
        with torch.no_grad():
            last_obs_tensor = transitions.next_obs[-1].to(self.device)
            last_q = model(last_obs_tensor).max(-1).values
            lambda_targets = self.compute_lambda_targets_fixed(
                last_q=last_q,
                q_vals=transitions.q_val.to(self.device),
                rewards=transitions.reward.to(self.device),
                dones=transitions.done.to(self.device),
                gamma=gamma,
                lam=lam,
            )

        # NETWORKS UPDATE
        total_loss = 0
        total_qvals = 0

        # Set network to train mode for parameter updates
        model.train()

        for epoch in range(num_epochs):
            # Prepare minibatches
            batch_size = num_steps * num_envs
            minibatch_size = batch_size // num_minibatches

            # Flatten and shuffle
            obs_flat = transitions.obs.reshape(-1, *transitions.obs.shape[2:])
            action_flat = transitions.action.reshape(-1)
            targets_flat = lambda_targets.reshape(-1)

            indices = torch.randperm(batch_size)

            for i in range(num_minibatches):
                start_idx = i * minibatch_size
                end_idx = start_idx + minibatch_size
                minibatch_indices = indices[start_idx:end_idx]

                obs_batch = obs_flat[minibatch_indices].to(self.device)
                action_batch = action_flat[minibatch_indices].to(self.device)
                target_batch = targets_flat[minibatch_indices].to(self.device)

                # Update learning rate if linear decay is enabled
                if lr_linear_decay:
                    cur_lr = self.lr_scheduler(
                        step=train_state.grad_steps,
                        lr=lr,
                        lr_linear_decay=lr_linear_decay,
                        num_minibatches=num_minibatches,
                        num_epochs=num_epochs,
                        num_updates_decay=num_updates_decay,
                    )
                    for pg in optimizer.param_groups:
                        pg["lr"] = cur_lr

                # Forward pass
                q_vals = model(obs_batch)
                chosen_action_qvals = q_vals.gather(
                    1, action_batch.unsqueeze(1)
                ).squeeze(1)

                # Compute loss
                loss = 0.5 * F.mse_loss(chosen_action_qvals, target_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                total_loss += loss.item()
                total_qvals += chosen_action_qvals.mean().item()
                train_state.grad_steps += 1

        train_state.n_updates += 1

        # Prepare metrics
        metrics = {
            "env_step": train_state.timesteps,
            "update_steps": train_state.n_updates,
            "env_frame": train_state.timesteps * frame_stack,
            "grad_steps": train_state.grad_steps,
            "td_loss": total_loss / (num_epochs * num_minibatches),
            "qvals": total_qvals / (num_epochs * num_minibatches),
        }

        # Add episode metrics - separate train and test
        if episode_returns:
            metrics["episode_return"] = np.mean(episode_returns)
            metrics["episode_length"] = np.mean(episode_lengths)

        # Add test metrics if available
        if test_during_training and test_episode_returns:
            metrics["test/episode_return"] = np.mean(test_episode_returns)
            metrics["test/episode_length"] = np.mean(test_episode_lengths)

        return train_state, obs, metrics

    def _train(
        self,
        env: gym.Env,
        model: nn.Module,
        num_updates: int,
        num_epochs: int,
        num_minibatches: int,
        frame_stack: int,
        total_envs: int,
        num_envs: int,
        num_test_envs: int,
        num_steps: int,
        test_during_training: bool,
        eps_start: float,
        eps_finish: float,
        eps_decay: float,
        num_updates_decay: int,
        rew_scale: float,
        gamma: float,
        lam: float,
        lr: float,
        lr_linear_decay: bool,
        max_grad_norm: float,
        seed: int = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)

        model.to(self.device)
        optimizer = optim.RAdam(model.parameters(), lr=lr)
        train_state = CustomTrainState(
            model=model,
            optimizer=optimizer,
        )

        # Initialize environment
        current_obs = env.reset()

        # Extract obs from (obs, info) tuple
        if isinstance(current_obs, tuple):
            current_obs = current_obs[0]

        # Ensure proper shape for vectorized envs
        if len(current_obs.shape) == 3:  # Single env
            current_obs = current_obs[np.newaxis, ...]  # Add batch dimension

        # Training loop
        all_metrics = []
        start_time = time.time()

        for update in range(num_updates):
            train_state, obs, metrics = self._update_step(
                env=env,
                model=model,
                optimizer=optimizer,
                num_epochs=num_epochs,
                num_minibatches=num_minibatches,
                frame_stack=frame_stack,
                train_state=train_state,
                current_obs=current_obs,
                total_envs=total_envs,
                num_envs=num_envs,
                num_test_envs=num_test_envs,
                num_steps=num_steps,
                test_during_training=test_during_training,
                eps_start=eps_start,
                eps_finish=eps_finish,
                eps_decay=eps_decay,
                num_updates_decay=num_updates_decay,
                rew_scale=rew_scale,
                gamma=gamma,
                lam=lam,
                lr=lr,
                lr_linear_decay=lr_linear_decay,
                max_grad_norm=max_grad_norm,
            )
            all_metrics.append(metrics)

            # More detailed logging
            if update % self.logging_freq == 0:
                elapsed_time = time.time() - start_time
                progress = (update + 1) / num_updates * 100

                # Agent 클래스 total_steps 업데이트
                self.total_steps = train_state.timesteps

                # Best 모델 저장 체크 (최소 10 에피소드 이후)
                if len(self.episode_rewards) >= 10:
                    saved = self.check_and_save_best(model, step=train_state.timesteps)
                    if saved:
                        self.logger.info(
                            f"🎉 새로운 최고 성능 모델 저장! Update: {update}"
                        )

                # Calculate current epsilon
                current_eps = self.eps_scheduler(
                    step=train_state.n_updates,
                    eps_start=eps_start,
                    eps_finish=eps_finish,
                    eps_decay=eps_decay,
                    num_updates_decay=num_updates_decay,
                )

                # Format episode stats if available
                episode_info = ""
                if "episode_return" in metrics:
                    episode_info = f", Ep.Ret: {metrics['episode_return']:.2f}, Ep.Len: {metrics.get('episode_length', 0):.0f}"

                # Add test episode info if available
                test_info = ""
                if "test/episode_return" in metrics:
                    test_info = f", Test.Ret: {metrics['test/episode_return']:.2f}"

                # Calculate steps per second
                steps_per_sec = (
                    metrics["env_step"] / elapsed_time if elapsed_time > 0 else 0
                )

                if self.logger:
                    self.logger.info(
                        f"Update {update + 1}/{num_updates} ({progress:.2f}%), "
                        f"Steps: {metrics['env_step']}, "
                        f"Grad Steps: {train_state.grad_steps}, "
                        f"LR: {lr:.6f}, "
                        f"Eps: {current_eps:.4f}, "
                        f"TD Loss: {metrics['td_loss']:.4f}, "
                        f"QVals: {metrics['qvals']:.4f}, "
                        f"Time: {elapsed_time:.2f}s, "
                        f"Steps/Sec: {steps_per_sec:.2f} "
                        f"{episode_info} {test_info}"
                    )

            # More detailed logging every 50 updates
            if update % self.detailed_logging_freq == 0 and update > 0:
                recent_metrics = all_metrics[-self.detailed_logging_freq :]
                avg_loss = np.mean([m["td_loss"] for m in recent_metrics])
                avg_qvals = np.mean([m["qvals"] for m in recent_metrics])

                # Agent 클래스의 성능 정보 추가
                if len(self.episode_rewards) > 0:
                    mean_reward = self.get_mean_reward(last_n=100)
                    best_reward = self.best_mean_reward
                    total_episodes = len(self.episode_rewards)

                    if self.logger:
                        self.logger.info(
                            f"Recent {self.detailed_logging_freq} Updates - "
                            f"Avg TD Loss: {avg_loss:.4f}, "
                            f"Avg QVals: {avg_qvals:.4f}"
                        )
                        self.logger.info(
                            f"성능 현황 - "
                            f"평균 보상: {mean_reward:.3f}, "
                            f"최고 보상: {best_reward:.3f}, "
                            f"총 에피소드: {total_episodes}"
                        )
                else:
                    if self.logger:
                        self.logger.info(
                            f"Recent {self.detailed_logging_freq} Updates - "
                            f"Avg TD Loss: {avg_loss:.4f}, "
                            f"Avg QVals: {avg_qvals:.4f}"
                        )

                # Calculate episode return trend if available
                episode_returns = [
                    m.get("episode_return", 0)
                    for m in recent_metrics
                    if "episode_return" in m
                ]
                if self.logger and episode_returns:
                    avg_return = np.mean(episode_returns)
                    max_return = np.max(episode_returns)
                    min_return = np.min(episode_returns)
                    self.logger.info(
                        f"Recent Episode Returns - Avg: {avg_return:.2f}, "
                        f"Max: {max_return:.2f}, Min: {min_return:.2f}"
                    )

                # Test return statistics if available
                test_returns = [
                    m.get("test/episode_return", 0)
                    for m in recent_metrics
                    if "test/episode_return" in m
                ]
                if self.logger and test_returns:
                    test_avg_return = np.mean(test_returns)
                    test_max_return = np.max(test_returns)
                    test_min_return = np.min(test_returns)
                    self.logger.info(
                        f"Recent Test Returns - Avg: {test_avg_return:.2f}, "
                        f"Max: {test_max_return:.2f}, Min: {test_min_return:.2f}"
                    )
        return {
            "train_state": train_state,
            "current_obs": current_obs,
            "metrics": all_metrics,
        }

    def learn(self, config: dict):
        self.device = config["device"]
        env_name = config["env_name"]

        # Agent 클래스의 모델 저장 기능 설정
        self.setup_model_saving(
            save_dir=self.save_dir,
            model_name=config["model_name"],
            max_best_models=config["max_best_models"],
        )

        save_config_path = os.path.join(self.save_dir, "config.json")
        with open(save_config_path, "w") as f:
            json.dump(config, f, indent=4)

        # Agent 클래스의 시작 시간 기록
        self.start_time = time.time()

        total_timesteps = config["total_timesteps"]
        total_timesteps_decay = config["total_timesteps_decay"]
        num_envs = config["num_envs"]
        num_steps = config["num_steps"]
        num_epochs = config["num_epochs"]
        num_minibatches = config["num_minibatches"]
        test_during_training = config["test_during_training"]
        num_test_envs = config["num_test_envs"] if test_during_training else 0
        eps_start = config["eps_start"]
        eps_finish = config["eps_finish"]
        eps_decay = config["eps_decay"]
        rew_scale = config["rew_scale"]
        gamma = config["gamma"]
        lam = config["lam"]
        lr = config["lr"]
        lr_linear_decay = config["lr_linear_decay"]
        max_grad_norm = config["max_grad_norm"]
        using_async = config["using_async"]
        frame_stack = config["frame_stack"]
        frame_skip = config["frame_skip"]
        noop_max = config["noop_max"]
        screen_size = config["screen_size"]
        terminal_on_life_loss = config["terminal_on_life_loss"]
        grayscale_obs = config["grayscale_obs"]
        scale_obs = config["scale_obs"]
        seed = config["seed"]

        num_updates = total_timesteps // num_steps // num_envs
        num_updates_decay = total_timesteps_decay // num_steps // num_envs

        assert (
            num_steps * num_envs
        ) % num_minibatches == 0, "num_minibatches must divide num_steps * num_envs"

        batch_size = (num_steps * num_envs) // num_minibatches

        total_envs = num_envs + num_test_envs
        env = self.make_env_vector(
            num_envs=total_envs,
            using_async=using_async,
            env_id=env_name,
            frame_stack=frame_stack,
            frame_skip=frame_skip,
            noop_max=noop_max,
            screen_size=screen_size,
            terminal_on_life_loss=terminal_on_life_loss,
            grayscale_obs=grayscale_obs,
            scale_obs=scale_obs,
            seed=seed,
        )

        outs = self._train(
            env=env,
            model=self.model,
            num_updates=num_updates,
            num_epochs=num_epochs,
            num_minibatches=num_minibatches,
            frame_stack=frame_stack,
            total_envs=total_envs,
            num_envs=num_envs,
            num_test_envs=num_test_envs,
            num_steps=num_steps,
            test_during_training=test_during_training,
            eps_start=eps_start,
            eps_finish=eps_finish,
            eps_decay=eps_decay,
            num_updates_decay=num_updates_decay,
            rew_scale=rew_scale,
            gamma=gamma,
            lam=lam,
            lr=lr,
            lr_linear_decay=lr_linear_decay,
            max_grad_norm=max_grad_norm,
            seed=seed,
        )

        end_time = time.time()
        total_time = max(end_time - self.start_time, 1e-6)

        if self.logger:
            self.logger.info(
                f"Training completed. "
                f"Total Time: {total_time/3600:.2f} hours ({total_time/60:.1f} minutes), "
                f"Total Steps: {total_timesteps}, "
                f"Steps per Second: {total_timesteps / total_time:.2f}"
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

        total_steps = total_timesteps
        steps_per_second = total_steps / total_time
        if self.logger:
            self.logger.info(
                f"Total Steps: {total_steps}, "
                f"Steps per Second: {steps_per_second:.2f}"
            )

        # Agent 클래스의 최종 모델 저장 기능 사용
        self.save_final_model(self.model, step=self.total_steps)

        self.model.cpu()

    def predict(self, obs, deterministic: bool = False):
        """
        학습된 모델을 사용해서 행동 예측

        Args:
            obs: 관측값 (numpy array or torch tensor)
            deterministic: True면 greedy 행동, False면 epsilon-greedy 행동

        Returns:
            action: 예측된 행동 (int or numpy array)
            q_values: Q-값들 (optional, numpy array)
        """
        self.model.eval()

        with torch.no_grad():
            # 입력 전처리
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float()
            else:
                obs_tensor = obs.float()

            # 배치 차원 추가 (필요한 경우)
            if len(obs_tensor.shape) == 3:  # (C, H, W) -> (1, C, H, W)
                obs_tensor = obs_tensor.unsqueeze(0)
            elif len(obs_tensor.shape) == 2:  # (H, W) -> (1, 1, H, W)
                obs_tensor = obs_tensor.unsqueeze(0).unsqueeze(0)

            # 디바이스로 이동
            obs_tensor = obs_tensor.to(self.device)

            # Q-값 계산
            q_values = self.model(obs_tensor)

            if deterministic:
                # Greedy 행동 선택 (최고 Q-값)
                actions = torch.argmax(q_values, dim=-1)
            else:
                # Epsilon-greedy 행동 선택 (기본 epsilon=0.01)
                eps = 0.01
                actions = self.eps_greedy_exploration(q_values, eps)

            # CPU로 이동 및 numpy 변환
            actions_np = actions.cpu().numpy()
            q_values_np = q_values.cpu().numpy()

            # 단일 관측값인 경우 스칼라 반환
            if obs_tensor.shape[0] == 1:
                return actions_np[0], q_values_np[0]
            else:
                return actions_np, q_values_np

    def predict_batch(self, obs_batch, deterministic: bool = False):
        """
        배치 관측값에 대해 행동 예측 (predict의 배치 버전)

        Args:
            obs_batch: 배치 관측값 (numpy array or torch tensor)
            deterministic: True면 greedy 행동, False면 epsilon-greedy 행동

        Returns:
            actions: 예측된 행동들 (numpy array)
            q_values: Q-값들 (numpy array)
        """
        return self.predict(obs_batch, deterministic=deterministic)

    def get_q_values(self, obs):
        """
        관측값에 대한 Q-값들만 반환

        Args:
            obs: 관측값 (numpy array or torch tensor)

        Returns:
            q_values: Q-값들 (numpy array)
        """
        _, q_values = self.predict(obs, deterministic=True)
        return q_values

    def load_model(self, model_path: str):
        """
        저장된 모델 가중치 로드

        Args:
            model_path: 모델 파일 경로 (.pth 파일)
        """
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.logger.info(f"모델 로드 완료: {model_path}")
        except Exception as e:
            self.logger.error(f"모델 로드 실패: {e}")
            raise e
