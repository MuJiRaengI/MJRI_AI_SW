import os
import cv2
import numpy as np
import pygame
import gymnasium as gym
import gymnasium.spaces as spaces
from typing import Optional, Tuple, Dict, Any

# ----------------- 색상 팔레트 -----------------
TILE_COLORS = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
    4096: (125, 226, 209),
    8192: (89, 195, 195),
    16384: (79, 134, 198),
    32768: (129, 90, 192),
    65536: (199, 44, 65),
    131072: (255, 106, 213),
}
FG_DARK = (119, 110, 101)
FG_LIGHT = (249, 246, 242)
BG_BOARD = (187, 173, 160)

# ------------------------------------------------


class Env2048PG(gym.Env):
    """
    Pygame 렌더링을 포함한 2048 Gymnasium 환경 (EnvAvoidObserver 스타일).

    액션:
      0: ↑(Up), 1: →(Right), 2: ↓(Down), 3: ←(Left) [Arrow Keys]

    관측(observation):
      Dict({
        "board":  (size, size) int32   - 타일 값 (0, 2, 4, ...)
        "vector": (4,) int32            - 유효 이동(one-hot; [Up, Right, Down, Left])
      })

    step 반환:
      obs, reward, done, info   # (당신의 기존 프로젝트 스타일과 일치)
    """

    def __init__(
        self,
        size: int = 4,
        max_exp: int = 16,
        seed: Optional[int] = None,
        reward_mode: str = "merge_sum",  # {"merge_sum","score_delta"}
        invalid_move_penalty: float = 0.0,
        target_tile: int = 16384,
        end_on_win: bool = False,
        spawn_prob_4: float = 0.1,
        max_tile_exp: int = 17,
        # 렌더링/UX
        tile_px: int = 100,
        gap_px: int = 10,
        fps: int = 30,
    ):
        super().__init__()
        assert size >= 2
        assert reward_mode in {"merge_sum", "custom", "score_delta"}
        assert 0.0 <= spawn_prob_4 <= 1.0

        self.size = size
        self.max_exp = max_exp
        self.reward_mode = reward_mode
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.target_tile = int(target_tile) if target_tile else 0
        self.end_on_win = bool(end_on_win)
        self.spawn_prob_4 = float(spawn_prob_4)
        self.max_tile_exp = int(max_tile_exp)
        self.tile_px = int(tile_px)
        self.gap_px = int(gap_px)
        self.fps = int(fps)

        # RNG
        self._rng = np.random.RandomState(None)
        # if seed is not None:
        #     self.seed(seed)

        # Spaces
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.max_exp + 1, self.size, self.size),
            dtype=np.int32,
        )

        # 상태
        self.board = np.zeros((self.size, self.size), dtype=np.int32)
        self.score = 0
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None

        # reward 가중치
        self.merged_value_weight = 0.1 * 10
        self.zero_rate_weight = 0.1 * 10
        self.max_tile_weight = 0.01 * 100
        self.pos_penalty_weight = 1.0

        # pygame/녹화
        self.screen = None
        self.clock = None
        self.font = None
        self.record = False
        self.video_writer = None

        # 캐시: 창 크기
        self._compute_window_size()

        # 초기화
        self.reset()

    # ---------------- Gym API ----------------
    def seed(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ):
        # if seed is not None:
        #     self.seed(seed)
        self.board.fill(0)
        self.score = 0
        self.steps = 0
        self.done = False
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None
        self._spawn_random_tile()
        self._spawn_random_tile()
        return self._get_obs(), None

    def calc_reward(self, merged_value) -> float:

        merged_score = np.log2(merged_value + 1) if merged_value > 0 else 0

        zero_rate = np.sum(self.board == 0) / 16

        max_tile = np.log2(np.max(self.board))

        max_pos = np.argwhere(self.board == max_tile)
        if max_pos.size > 0:
            y, x = max_pos[0]
            pos_penalty = (x + y) / (2 * self.size - 2)
        else:
            pos_penalty = 0.0

        # apply weights
        merged_score *= self.merged_value_weight
        zero_rate *= self.zero_rate_weight
        max_tile *= self.max_tile_weight
        pos_penalty *= self.pos_penalty_weight

        reward = merged_score + zero_rate + max_tile - pos_penalty
        return reward

    def step(self, action: int):
        if self.done:
            print(f"max value : {self.board.max()}")
            return (
                self._get_obs(),
                0.0,
                True,
                False,
                {"score": self.score, "won": self._won()},
            )

        assert self.action_space.contains(action), "Invalid action"
        prev_score = self.score
        merged_value, changed = self._move(action)

        # (유효한 이동이 하나도 없으면 종료)
        if not changed:
            reward = float(self.invalid_move_penalty)
            self.reward = reward
            self.total_reward += reward
            self.steps += 1
            self.last_action = action
            done = False
            # 종료 체크 (유효한 이동이 하나도 없으면 종료)
            if not self._any_moves_left():
                done = True
                self.done = True
            return (
                self._get_obs(),
                reward,
                False,
                False,
                {
                    "score": self.score,
                    "won": self._won(),
                    "invalid": True,
                },
            )

        # 유효 이동 → 새 타일 스폰
        self._spawn_random_tile()

        # 보상
        if self.reward_mode == "merge_sum":
            reward = float(merged_value)
        elif self.reward_mode == "custom":
            # reward = np.log2(float(merged_value) + 1)
            reward = self.calc_reward(merged_value)
        else:
            reward = float(self.score - prev_score)

        won = self._won()
        no_moves = not self._any_moves_left()
        terminated = (self.end_on_win and won) or no_moves
        truncated = False
        self.done = terminated or truncated

        self.reward = reward
        self.total_reward += reward
        self.steps += 1
        self.last_action = action

        info = {"score": self.score, "won": won, "no_moves": no_moves}
        return self._get_obs(), reward, terminated, truncated, info

    # ------------- Helper: observation --------------
    def _get_obs(self):
        return {
            "board": (self.board.copy().astype(np.int32),),
            "vector": self._valid_moves_one_hot().astype(np.int32),
        }
        # return (self.board.copy().astype(np.int32),)

    def _valid_moves_one_hot(self):
        onehot = np.zeros(4, dtype=np.int32)
        for a in range(4):
            changed = self._peek_change(a)
            onehot[a] = 1 if changed else 0
        return onehot

    def _peek_change(self, action: int) -> bool:
        # 보드/점수 백업 → _move → 복구
        board_bk = self.board.copy()
        score_bk = self.score
        merged_val, changed = self._move(action)
        # 복구
        self.board = board_bk
        self.score = score_bk
        return changed

    # ------------- Mechanics -----------------
    def _won(self) -> bool:
        return self.target_tile > 0 and int(self.board.max()) >= self.target_tile

    def _spawn_random_tile(self) -> None:
        empties = list(zip(*np.where(self.board == 0)))
        if not empties:
            return
        r, c = empties[self._rng.randint(len(empties))]
        self.board[r, c] = 4 if self._rng.rand() < self.spawn_prob_4 else 2

    def _any_moves_left(self) -> bool:
        b = self.board
        if np.any(b == 0):
            return True
        if np.any(b[:, :-1] == b[:, 1:]):
            return True
        if np.any(b[:-1, :] == b[1:, :]):
            return True
        return False

    def _move(self, action: int) -> Tuple[int, bool]:
        """좌/우/상/하 이동 처리. (merged_sum, changed) 반환. self.score 갱신."""
        rotated = self._rotate_for_action(self.board, action)
        merged_value = 0
        changed_any = False
        new_rows = []

        for r in range(self.size):
            row = rotated[r].copy()
            nz = row[row != 0]
            merged = []
            i = 0
            while i < len(nz):
                if i + 1 < len(nz) and nz[i] == nz[i + 1]:
                    val = nz[i] * 2
                    merged.append(val)
                    merged_value += val
                    self.score += val
                    i += 2
                else:
                    merged.append(nz[i])
                    i += 1
            merged_arr = np.array(merged, dtype=np.int32)
            if merged_arr.size < self.size:
                merged_arr = np.pad(merged_arr, (0, self.size - merged_arr.size))
            new_rows.append(merged_arr)
            if not np.array_equal(row, merged_arr):
                changed_any = True

        new_rotated = np.vstack(new_rows)
        new_board = self._inv_rotate_for_action(new_rotated, action)
        if changed_any:
            self.board = new_board
        return merged_value, changed_any

    @staticmethod
    def _rotate_for_action(board: np.ndarray, action: int) -> np.ndarray:
        a = int(action) % 4
        if a == 0:  # Up -> +90° CCW
            return np.rot90(board, 1)
        if a == 1:  # Right -> 180°
            return np.rot90(board, 2)
        if a == 2:  # Down -> -90° CW
            return np.rot90(board, -1)
        return board.copy()  # Left -> no rotation

    @staticmethod
    def _inv_rotate_for_action(board: np.ndarray, action: int) -> np.ndarray:
        a = int(action) % 4
        if a == 0:
            return np.rot90(board, -1)
        if a == 1:
            return np.rot90(board, 2)
        if a == 2:
            return np.rot90(board, 1)
        return board.copy()

    # ------------- Pygame Render -------------
    def _compute_window_size(self):
        size = self.size
        grid_w = size * self.tile_px + (size + 1) * self.gap_px
        grid_h = grid_w
        # 상단 정보 영역 여유
        self.window_w = grid_w + 40
        self.window_h = grid_h + 120

    def render(self, mode="human", logits=None):
        try:
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode((self.window_w, self.window_h))
                pygame.display.set_caption("2048 — pygame")
                self.clock = pygame.time.Clock()
                self.font = pygame.font.SysFont("Arial", 24)
                self.font_big = pygame.font.SysFont("Arial", 36, bold=True)

            self.screen.fill((250, 248, 239))
            # 제목/점수
            head_y = 20
            self._blit_text(
                f"2048 — Arrow Keys | R: Restart | A: Autoplay",
                20,
                head_y,
                FG_DARK,
                self.font,
            )
            self._blit_text(f"Score: {self.score}", 20, head_y + 30, FG_DARK, self.font)

            # 보드 영역
            grid_x = 20
            grid_y = 80
            grid_w = self.window_w - 40
            # 보드 배경
            pygame.draw.rect(
                self.screen,
                BG_BOARD,
                (grid_x, grid_y, grid_w, grid_w),
                border_radius=10,
            )

            # 셀 그리기
            size = self.size
            cell = self.tile_px
            gap = self.gap_px
            for r in range(size):
                for c in range(size):
                    val = int(self.board[r, c])
                    color = TILE_COLORS.get(val, (60, 58, 50))
                    x = grid_x + gap + c * (cell + gap)
                    y = grid_y + gap + r * (cell + gap)
                    pygame.draw.rect(
                        self.screen, color, (x, y, cell, cell), border_radius=8
                    )

                    if val > 0:
                        fg = FG_DARK if val <= 4 else FG_LIGHT
                        # 셀 크기에 맞춰 폰트 크기 조절
                        fsize = max(18, int(cell * 0.35))
                        font_dyn = pygame.font.SysFont("Arial", fsize, bold=True)
                        self._blit_center(str(val), x, y, cell, cell, fg, font_dyn)

            # 게임 종료 오버레이
            if self.done:
                overlay = pygame.Surface((grid_w, grid_w), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 120))
                self.screen.blit(overlay, (grid_x, grid_y))
                msg = "You Win!" if self._won() else "Game Over"
                self._blit_center(
                    msg, grid_x, grid_y, grid_w, grid_w, FG_LIGHT, self.font_big
                )

            pygame.display.flip()
            if self.clock:
                self.clock.tick(self.fps)

            # 녹화
            if self.record and self.video_writer is not None:
                frame = pygame.surfarray.array3d(self.screen)
                frame = frame.swapaxes(0, 1)  # (W,H,C) -> (H,W,C)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame)
        except Exception:
            # 렌더 관련 오류는 조용히 무시(당신의 기존 스타일과 동일)
            pass

    def _blit_text(self, text, x, y, color, font):
        surf = font.render(text, True, color)
        self.screen.blit(surf, (x, y))

    def _blit_center(self, text, x, y, w, h, color, font):
        surf = font.render(text, True, color)
        rect = surf.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(surf, rect)

    # ------------- Recording -------------
    def start_recording(self, filename="record_2048.mp4", fps: Optional[int] = None):
        if fps is None:
            fps = self.fps
        self.record = True
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(
            filename, fourcc, fps, (self.window_w, self.window_h)
        )

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
        self.video_writer = None
        self.record = False

    # ------------- Utils -------------
    def close(self):
        try:
            if self.video_writer:
                self.video_writer.release()
            if self.screen is not None:
                pygame.quit()
        except Exception:
            pass


# --------------------- 간단 실행기 ---------------------
def _autoplay_step(env: Env2048PG):
    # 간단 휴리스틱: Down, Left, Right, Up 순서로 가능한 첫 동작 수행
    for a in (2, 3, 1, 0):
        if env._peek_change(a):
            return a
    return None  # 더 이상 이동 불가


if __name__ == "__main__":
    env = Env2048PG(size=4)

    obs = env.reset()
    done = False
    autoplay = False
    action = None

    # 최초 렌더
    env.render()

    while not done:
        # 이벤트 처리
        for event in pygame.event.get():
            if done:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        done = True
                        break
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        done = False
                        action = None
                        break
                elif event.type == pygame.QUIT:
                    done = True
                    break
                continue  # Ignore all other events when done
            else:
                if event.type == pygame.QUIT:
                    done = True
                elif event.type == pygame.KEYDOWN:
                    if event.key in (
                        pygame.K_UP,
                        pygame.K_RIGHT,
                        pygame.K_DOWN,
                        pygame.K_LEFT,
                    ):
                        key2act = {
                            pygame.K_UP: 0,
                            pygame.K_RIGHT: 1,
                            pygame.K_DOWN: 2,
                            pygame.K_LEFT: 3,
                        }
                        action = key2act[event.key]
                    elif event.key == pygame.K_r:
                        obs = env.reset()
                        done = False
                        action = None
                    elif event.key == pygame.K_a:
                        autoplay = not autoplay

        # 오토플레이
        if autoplay and not done:
            a = _autoplay_step(env)
            action = a if a is not None else None

        # 스텝
        if action is not None and not done:
            obs, reward, done, info = env.step(action)
            action = None

        env.render()

    env.stop_recording()
    env.close()
