from collections import deque
from itertools import chain
import tqdm
import copy
import os
from multiprocessing import freeze_support

import numpy as np

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from .experience_replay import *
from .utils import *

import locale
import keyboard

locale.getpreferredencoding = lambda: "UTF-8"


class BBF:
    def __init__(
        self,
        model,
        env,
        memory_size=50000,
        learning_rate=1e-4,
        batch_size=32,
        ema_decay=0.995,
        initial_gamma=0.97,
        final_gamma=0.997,
        initial_n=10,
        final_n=3,
        num_buckets=51,
        reset_freq=40000,
        replay_ratio=2,
        weight_decay=0.1,
        epsilon=0,
        gym_env=False,
        stackFrame=True,
        real_env=False,
    ):

        self.model = model
        self.model_target = copy.deepcopy(model)
        self.env = env
        self.memory_size = memory_size
        self.n_actions = env.action_space.n
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.ema_decay = ema_decay
        self.initial_gamma = torch.tensor(1 - initial_gamma).log()
        self.final_gamma = torch.tensor(1 - final_gamma).log()
        self.initial_n = initial_n
        self.final_n = final_n
        self.num_buckets = num_buckets
        self.reset_freq = reset_freq
        self.schedule_max_step = reset_freq // 4
        self.replay_ratio = replay_ratio
        self.weight_decay = weight_decay
        self.epsilon = epsilon
        self.transforms = transforms.Compose([transforms.Resize((96, 72))])
        self.gym_env = gym_env
        self.stackFrame = stackFrame
        self.real_env = real_env
        self.running = True
        keyboard.add_hotkey("ctrl+p", self.stop_running)
        keyboard.add_hotkey("f4", self.stop_running)

    def stop_running(self):
        self.running = False

    def learn(
        self,
        total_timesteps,
        save_freq=None,
        save_path=None,
        name_prefix="bbf",
    ):

        perception_modules = [self.model.encoder_cnn, self.model.transition]
        actor_modules = [
            self.model.prediction,
            self.model.projection,
            self.model.a,
            self.model.v,
        ]

        params_wm = []
        for module in perception_modules:
            for param in module.parameters():
                if param.requires_grad == True:  # They all require grad
                    params_wm.append(param)

        params_ac = []
        for module in actor_modules:
            for param in module.parameters():
                if param.requires_grad == True:
                    params_ac.append(param)

        self.optimizer = torch.optim.AdamW(
            chain(params_wm, params_ac),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            eps=1.5e-4,
        )

        self.memory = PrioritizedReplay_nSteps_Sqrt(self.memory_size)
        self.memory.free()

        scores = []
        step = 0
        grad_step = 0

        progress_bar = tqdm.tqdm(total=total_timesteps)

        if self.gym_env:
            state = self.env.reset()
        else:
            state, info = self.env.reset()
        state = self.model.preprocess(state).unsqueeze(0)

        if self.stackFrame:
            states = deque(maxlen=4)
            for i in range(4):
                states.append(state)

        eps_reward = torch.tensor([0], dtype=torch.float)

        reward = np.array([0])
        done_flag = np.array([False])
        terminated = np.array([False])

        last_lives = np.array([0])
        life_loss = np.array([0])
        resetted = np.array([0])

        last_grad_update = 0
        episode_steps = 0
        while step < total_timesteps:
            if not self.running:
                break
            # print(f"step {step}, grad_step {grad_step}")
            episode_steps += 1
            progress_bar.update(1)
            self.model_target.train()

            # len_memory = len(self.memory)

            # if resetted[0]>0:
            #    states = env.noop_steps(states)

            if self.stackFrame:
                Q_action = self.model_target.env_step(
                    torch.cat(list(states), -3).unsqueeze(0)
                )
            else:
                Q_action = self.model_target.env_step(state.unsqueeze(0))

            action = epsilon_greedy(
                Q_action, self.n_actions, len(self.memory), self.epsilon
            ).cpu()

            if self.stackFrame:
                self.memory.push(
                    torch.cat(list(states), -3).detach().cpu(),
                    torch.tensor(reward, dtype=torch.float),
                    action,
                    torch.tensor(np.logical_or(done_flag, life_loss), dtype=torch.bool),
                )
            else:
                self.memory.push(
                    state.detach().cpu(),
                    torch.tensor(reward, dtype=torch.float),
                    action,
                    torch.tensor(np.logical_or(done_flag, life_loss), dtype=torch.bool),
                )
            # print('action', action, action.shape)

            len_memory = len(self.memory)

            if self.gym_env:
                state, reward, terminated, info = self.env.step(action.item())
                truncated = False
            else:
                state, reward, terminated, truncated, info = self.env.step(
                    action.item()
                )

            state = self.model.preprocess(state).unsqueeze(0)
            if self.stackFrame:
                states.append(state)
            reward = np.array([reward])
            terminated = np.array([terminated])
            truncated = np.array([truncated])

            eps_reward += reward
            reward = np.clip(reward, -1, 1)

            done_flag = np.logical_or(terminated, truncated)
            if "lives" in info:
                lives = np.array([info["lives"]])
                life_loss = np.clip(last_lives - lives, 0, None)
                resetted = np.clip(lives - last_lives, 0, None)
                last_lives = lives

            n = int(
                self.initial_n
                * (self.final_n / self.initial_n)
                ** (min(grad_step, self.schedule_max_step) / self.schedule_max_step)
            )
            n = np.array(n).item()

            self.memory.priority[len_memory - 1] = self.memory.max_priority()

            if (
                not self.real_env
                and len_memory > self.batch_size
                and len_memory > self.memory_size // 2
            ):
                for i in range(self.replay_ratio):
                    self.optimize(grad_step, n)
                    target_model_ema(self.model, self.model_target)
                    grad_step += 1

            if self.real_env and done_flag:
                for _ in tqdm.tqdm(range(episode_steps)):
                    if (
                        len_memory > self.batch_size
                        and len_memory > self.memory_size // 2
                    ):
                        for i in range(self.replay_ratio):
                            self.optimize(grad_step, n)
                            target_model_ema(self.model, self.model_target)
                            grad_step += 1
                print(f"Episode finished after {episode_steps} steps")
                episode_steps = 0

            if save_freq != None and ((step + 1) % save_freq) == 0:
                save_checkpoint_path = f"{save_path}/{name_prefix}_{step+1}_steps.tmp"
                save_checkpoint(self.model, self.model_target, save_checkpoint_path)
                # renaming
                try:
                    os.replace(
                        save_checkpoint_path,
                        f"{save_path}/{name_prefix}_{step+1}_steps.pth",
                    )
                    print(
                        f"[INFO] 모델 체크포인트 저장 완료: {save_path}/{name_prefix}_{step+1}_steps.pth"
                    )
                except Exception as e:
                    print(f"[ERROR] 파일 교체 실패: {e}")

            if grad_step > self.reset_freq:
                # eval()
                print("Reseting on step", step, grad_step)

                # seed_np_torch(random.randint(SEED-1000, SEED+1000)+step)
                self.model.hard_reset()

                # seed_np_torch(random.randint(SEED-1000, SEED+1000)+step)
                self.model_target.hard_reset()

                grad_step = 0

                actor_modules = [
                    self.model.prediction,
                    self.model.projection,
                    self.model.a,
                    self.model.v,
                ]
                params_ac = []
                for module in actor_modules:
                    for param in module.parameters():
                        params_ac.append(param)

                perception_modules = [self.model.encoder_cnn, self.model.transition]
                params_wm = []
                for module in perception_modules:
                    for param in module.parameters():
                        params_wm.append(param)

                optimizer_aux = torch.optim.AdamW(
                    params_wm,
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    eps=1.5e-4,
                )
                copy_states(self.optimizer, optimizer_aux)
                self.optimizer = torch.optim.AdamW(
                    chain(params_wm, params_ac),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay,
                    eps=1.5e-4,
                )
                copy_states(optimizer_aux, self.optimizer)

            step += 1

            log_t = done_flag.astype(float).nonzero()[0]

            if len(log_t) > 0:
                for log in log_t:
                    # print({"eps_reward": eps_reward[log].sum()})
                    scores.append(eps_reward[log].clone())

                eps_reward[log_t] = 0
                if self.gym_env:
                    state = self.env.reset()
                else:
                    state, info = self.env.reset()
                state = self.model.preprocess(state).unsqueeze(0)

                if self.stackFrame:
                    states = deque(maxlen=4)
                    for i in range(4):
                        states.append(state)

        save_final_path = f"{save_path}/{name_prefix}.tmp"
        save_checkpoint(self.model, self.model_target, save_final_path)
        # renaming
        try:
            os.replace(
                save_final_path,
                f"{save_path}/{name_prefix}.pth",
            )
        except Exception as e:
            print(f"[ERROR] 파일 교체 실패: {e}")

    def optimize(self, grad_step, n):
        self.model.train()
        self.model_target.train()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=False):
            with torch.no_grad():
                states, next_states, rewards, action, c_flag, idxs, is_w = (
                    self.memory.sample(n, self.batch_size, grad_step)
                )
                z = self.model_target.encode(states[:, 1:6])[0]

            terminal = 1 - c_flag
            # print(f"STUFF HERE {states.shape, rewards.shape, c_flag.shape, action.shape, n}")

            q, max_action, _, z_pred = self.model(
                states[:, 0][:, None], action[:, :5].long()
            )

            max_action = self.model.get_max_action(next_states[:, n - 1][:, None])
            next_values = self.model_target.evaluate(
                next_states[:, n - 1][:, None].contiguous(), max_action
            )

            action = action[:, 0, None].expand(self.batch_size, self.num_buckets)
            action = action[:, None]
            with torch.no_grad():
                gammas_one = torch.ones(
                    self.batch_size, n, 1, dtype=torch.float, device="cuda"
                )
                gamma_step = (
                    1
                    - torch.tensor(
                        (
                            (
                                self.schedule_max_step
                                - min(grad_step, self.schedule_max_step)
                            )
                            / self.schedule_max_step
                        )
                        * (self.initial_gamma - self.final_gamma)
                        + self.final_gamma
                    ).exp()
                )
                gammas = gammas_one * gamma_step

                returns = []
                for t in range(n):
                    ret = 0
                    for u in reversed(range(t, n)):
                        # print(c_flag.shape, gammas.shape, rewards.shape)
                        ret += (
                            torch.prod(c_flag[:, t + 1 : u + 1], -2)
                            * torch.prod(gammas[:, t:u], -2)
                            * rewards[:, u + 1]
                        )
                    returns.append(ret)
                returns = torch.stack(returns, 1)

            plot_vs = returns.clone().sum(-1)

            same_traj = (torch.prod(c_flag[:, :n], -2)).squeeze()

            returns = returns[:, 0]
            returns = (
                returns
                + torch.prod(gammas[0, :10], -2).squeeze()
                * same_traj[:, None]
                * self.model.support[None, :]
            )
            returns = returns.squeeze()

            next_values = next_values[:, 0]

            log_probs = torch.log(
                q[:, 0].gather(-2, action)[:, None] + eps
            ).contiguous()

            dist = project_distribution(
                returns, next_values.squeeze(), self.model.support
            )

            loss = (
                -(dist * (log_probs.squeeze()))
                .sum(-1)
                .view(self.batch_size, -1)
                .sum(-1)
            )
            dqn_loss = loss.clone().mean()
            td_error = (
                loss + torch.nan_to_num((dist * torch.log(dist))).sum(-1)
            ).mean()

            batched_loss = loss.clone()

            z = F.normalize(z, 2, dim=-1, eps=1e-5)
            z_pred = F.normalize(z_pred, 2, dim=-1, eps=1e-5)

            recon_loss = torch.nn.functional.mse_loss(
                z_pred.contiguous().view(-1, 2048),
                z.contiguous().view(-1, 2048),
                reduction="none",
            ).sum(-1)
            recon_loss = 5 * (recon_loss.view(self.batch_size, -1).mean(-1)) * same_traj

            loss += recon_loss

            loss = (loss * is_w).mean()  # mean across batch axis

        loss.backward()

        param_norm, grad_norm = params_and_grad_norm(self.model)
        # scaler.scale(loss).backward()
        # scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
        # scaler.step(optimizer)
        # scaler.update()

        self.optimizer.step()
        self.optimizer.zero_grad()

        # memory.set_priority(idxs, batched_loss)
        self.memory.set_priority(idxs, batched_loss, same_traj)

        lr = self.optimizer.param_groups[0]["lr"]

        # print(
        #     f"Step: {grad_step}, Loss: {loss.item():.4f}, DQN Loss: {dqn_loss.item():.4f}, "
        #     f"Recon Loss: {recon_loss.mean().item():.4f}, LR: {lr:.6f}, "
        #     f"Returns: {plot_vs.mean().item():.4f}, Buffer Rewards: {rewards.mean(0).sum().item():.4f}, "
        #     f"Is W: {is_w.mean().item():.4f}, Gamma: {gamma_step.item():.4f}, "
        #     f"TD Error: {td_error.item():.4f}, Param Norm: {param_norm.sum().item():.4f}, "
        #     f"Grad Norm: {grad_norm.sum().item():.4f}"
        # )

    def save(self, save_path):
        save_checkpoint(self.model, self.model_target, save_path)

    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path)["model_state_dict"])
        self.model_target.load_state_dict(
            torch.load(load_path)["model_target_state_dict"]
        )
