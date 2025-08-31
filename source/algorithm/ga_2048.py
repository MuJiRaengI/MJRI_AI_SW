import os
import sys

sys.path.append(os.path.abspath("."))
import json
import time
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import numpy as np

from source.algorithm.algorithm import Algorithm
from source.envs.env2048 import Env2048PG


class GA2048(Algorithm):
    def __init__(
        self,
        size: int,
        save_dir: str,
        logging_freq=100,
        detailed_logging_freq=500,
    ):
        super().__init__(save_dir, logging_freq, detailed_logging_freq)
        self.size = size
        self.env = Env2048PG(self.size)
        self.history = {}

    # --------------------------
    def learn(self, config: dict):
        # 0) 설정 저장
        os.makedirs(self.save_dir, exist_ok=True)
        save_config_path = os.path.join(self.save_dir, "config.json")
        with open(save_config_path, "w") as f:
            json.dump(config, f, indent=4)

        self.start_time = time.time()

        # 1) 설정 파싱 & 시드 고정
        seed = config["seed"]
        if seed is not None and seed >= 0:
            random.seed(seed)
            np.random.seed(seed)

        # 필수 파라미터 읽기 (전부 dict['key'])
        gene_dim = config["gene_dim"]
        weight_init_sigma = config["weight_init_sigma"]

        eval_episodes = config["eval_episodes"]
        max_steps = config["max_steps"]
        bonus_2048 = config["bonus_2048"]
        invalid_move_penalty = config["invalid_move_penalty"]

        population_size = config["population_size"]
        generations = config["generations"]
        elitism = config["elitism"]
        tournament_k = config["tournament_k"]
        crossover_rate = config["crossover_rate"]
        mutation_p = config["mutation_p"]
        mutation_sigma_start = config["mutation_sigma_start"]
        mutation_sigma_end = config["mutation_sigma_end"]
        mutation_sigma_decay = config["mutation_sigma_decay"]

        save_best_every = config["save_best_every"]
        eval_final_episodes = config["eval_final_episodes"]

        # 2) 초기 개체군
        pop: List[np.ndarray] = [
            (np.random.randn(gene_dim).astype(np.float32) * weight_init_sigma)
            for _ in range(population_size)
        ]

        self.history = {
            # "cfg": config,
            "gens": [],
        }

        # 3) GA 루프
        for g in range(generations):
            # 3-1) 평가
            fits = []
            for w in tqdm(pop, desc=f"Gen {g+1}/{generations} Eval"):
                fit = self.fitness(
                    w,
                    eval_episodes,
                    max_steps,
                    bonus_2048,
                    invalid_move_penalty,
                )
                fits.append(fit)

            elite_indices = np.argsort(fits)[-elitism:][::-1].tolist()
            best_idx = int(elite_indices[0])
            best_fit = float(fits[best_idx])
            best_w = pop[best_idx].copy()

            # 로깅
            self.logger.info(
                f"[Gen {g}] best_fit={best_fit:.2f} avg_fit={float(np.mean(fits)):.2f}"
            )

            # 3-2) 세대 교체
            new_pop: List[np.ndarray] = []
            # 엘리트 보존
            for ei in elite_indices:
                new_pop.append(pop[ei].copy())

            # 변이 강도 스케줄
            sigma = max(
                mutation_sigma_start * (mutation_sigma_decay**g), mutation_sigma_end
            )

            # 나머지 채우기
            while len(new_pop) < population_size:
                p1 = self.tournament_select(pop, fits, tournament_k)
                p2 = self.tournament_select(pop, fits, tournament_k)
                child = self.crossover(p1, p2, crossover_rate)
                child = self.mutate(child, sigma=sigma, p=mutation_p)
                new_pop.append(child.astype(np.float32))

            pop = new_pop

            # 3-3) 기록/저장
            gen_rec = {
                "gen": g,
                "best_fit": best_fit,
                "avg_fit": float(np.mean(fits)),
                "sigma": sigma,
            }
            self.history["gens"].append(gen_rec)

            if (g % save_best_every) == 0:
                self._save_checkpoint(g, best_w, best_fit)

        # 4) 최종 베스트 선택 (엄격 재평가)
        config_final = dict(config)
        config_final["eval_episodes"] = eval_final_episodes

        final_fits = [
            self.fitness(
                w,
                config_final["eval_episodes"],
                max_steps,
                bonus_2048,
                invalid_move_penalty,
            )
            for w in pop
        ]
        final_idx = int(np.argmax(final_fits))
        final_best_w = pop[final_idx]
        final_best_fit = float(final_fits[final_idx])

        # 최종 저장
        np.savez_compressed(
            os.path.join(self.save_dir, "best_final.npz"),
            weights=final_best_w.astype(np.float32),
            fitness=np.array([final_best_fit], dtype=np.float32),
        )
        self._save_history()
        self.logger.info(
            f"[Done] final_best_fit={final_best_fit:.2f} saved to {self.save_dir}"
        )

    # --------------------------
    def policy_forward(self, board: np.ndarray, w: np.ndarray) -> int:
        valid_actions = [0, 1, 2, 3]
        return random.choice(valid_actions)

    def rollout(
        self, w: np.ndarray, max_steps: int, invalid_move_penalty: float, seed: int = 0
    ) -> Tuple[float, int]:
        random.seed(seed)
        np.random.seed(seed)
        obs, _ = self.env.reset(seed=seed)
        board = obs["board"][0]
        total = 0.0
        max_tile = 0
        dead_steps = 0

        for _ in range(max_steps):
            action = self.policy_forward(board, w)
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            next_board = next_obs["board"][0]
            if (next_board == board).all():
                if invalid_move_penalty != 0.0:
                    total += invalid_move_penalty
                dead_steps += 1
                if dead_steps >= 3:
                    break
            else:
                dead_steps = 0

            board = next_board
            total += float(reward)
            max_tile = max(max_tile, int(board.max()))
            done = terminated or truncated
            if done:
                break

        return total, max_tile

    def fitness(
        self,
        w: np.ndarray,
        eval_episodes: int,
        max_steps: int,
        bonus_2048: float,
        invalid_move_penalty: float,
    ) -> float:
        scores, tiles = [], []
        for ep in range(eval_episodes):
            s, mt = self.rollout(w, max_steps, invalid_move_penalty, seed=ep)
            scores.append(s)
            tiles.append(mt)
        bonus = bonus_2048 * sum(mt >= 2048 for mt in tiles)
        return float(np.mean(scores)) + bonus

    def tournament_select(
        self, pop: List[np.ndarray], fits: List[float], k: int
    ) -> np.ndarray:
        idxs = np.random.choice(len(pop), size=k, replace=False)
        best_idx = max(idxs, key=lambda i: fits[i])
        return pop[best_idx].copy()

    def crossover(self, p1: np.ndarray, p2: np.ndarray, rate: float) -> np.ndarray:
        mask = np.random.rand(*p1.shape) < rate
        child = p1.copy()
        child[mask] = p2[mask]
        return child

    def mutate(self, w: np.ndarray, sigma: float, p: float) -> np.ndarray:
        mask = np.random.rand(*w.shape) < p
        noise = np.random.randn(*w.shape) * sigma
        return w + mask * noise

    def _save_checkpoint(self, gen: int, best_w: np.ndarray, best_fit: float):
        ckpt_dir = os.path.join(self.save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        np.savez_compressed(
            os.path.join(ckpt_dir, f"gen_{gen:04d}.npz"),
            weights=best_w.astype(np.float32),
            fitness=np.array([best_fit], dtype=np.float32),
            gen=np.array([gen], dtype=np.int32),
        )

    def _save_history(self):
        with open(os.path.join(self.save_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=2)

    def predict(self):
        pass


if __name__ == "__main__":
    with open(
        r"C:\Users\stpe9\Desktop\vscode\MJRI_AI_SW\configs\algorithm\GA\2048\config.json",
        "r",
    ) as f:
        config = json.load(f)

    ga = GA2048(size=4, save_dir=config["save_dir"])
    ga.learn(config)
