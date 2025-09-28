# batch2048_core.py
from __future__ import annotations
import enum
import time
import threading

import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import VecEnv

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


class Batch2048Core:
    """
    벡터화 2048 순수 코어

    - 내부 상태: boards (N, 4) uint16, 각 원소는 4칸(=4니블, 상위→하위 칸)
            - 한 행(16비트)에 4칸의 지수 e(0..15)를 담는다. 실제 타일 값은 2^e, e=0은 빈칸.
    - 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    - 보상: 2048 룰(두 타일 e,e가 병합되어 e+1 생성 시 2^(e+1) 점수) 합
    - 스폰: 이동이 발생한 보드에만 1개 타일 스폰 (2: 확률 1-p4, 4: 확률 p4)
    - obs_mode:
            - "uint16x4" : (N, 4) uint16, 내부 보드 그대로
            - "uint8x16" : (N, 16) uint8, 16칸의 지수값 벡터
            - "onehot256": (N, 256) uint8/float32, 16칸×16클래스 원-핫 (flatten)
    """

    # -------- 클래스 정적 LUT들 (최초 1회 생성) --------
    _LUT_LEFT_NEW: np.ndarray | None = None  # uint16[65536], LEFT 적용 결과 행
    _LUT_RIGHT_NEW: np.ndarray | None = None  # uint16[65536], RIGHT 적용 결과 행
    _LUT_LEFT_MOV: np.ndarray | None = None  # bool[65536],  LEFT 적용 시 변화 여부
    _LUT_RIGHT_MOV: np.ndarray | None = None  # bool[65536],  RIGHT 적용 시 변화 여부
    _LUT_LR_NEW: np.ndarray | None = None  # uint16[65536, 2] (0=left,1=right)
    _LUT_LEFT_REW: np.ndarray | None = None  # uint32[65536], LEFT 보상 합
    _LUT_RIGHT_REW: np.ndarray | None = None  # uint32[65536], RIGHT 보상 합

    # 스폰 최적화용 LUT들
    _PC4: np.ndarray | None = None  # uint8[16],     4비트 popcount
    _PC16: np.ndarray | None = None  # uint16[65536], 16비트 popcount
    _LUT_EMPTY4_ROW: np.ndarray | None = None  # uint16[65536], row16 -> empty 4bit mask
    _LUT_MASK_SELECT: np.ndarray | None = (
        None  # uint8[16,4],   (mask4, nth) -> col or 255
    )
    _LUT_SELECT16_ROWS: np.ndarray | None = (
        None  # uint16[65536,16], (mask16,nth)->row or 255
    )
    _LUT_SELECT16_COLS_REVERSE: np.ndarray | None = (
        None  # uint16[65536,16], (mask16,nth)->col or 255
    )

    class ObsMode(enum.IntEnum):
        UINT16x4 = enum.auto()
        UINT8x16 = enum.auto()
        ONEHOT256 = enum.auto()

    def __init__(
        self,
        obs_mode: ObsMode = ObsMode.UINT8x16,
        num_envs: int = 1024,
        seed: int | None = None,
        p4: float = 0.1,
        render_mode: str | None = None,
    ):
        self.num_envs = int(num_envs)
        self.obs_mode = obs_mode
        self._obs_func = self._select_obs_fn(obs_mode)
        self._rng = np.random.default_rng(seed)
        self._boards = np.zeros((self.num_envs, 4), dtype=np.uint16)
        self._boards_T = np.zeros((self.num_envs, 4), dtype=np.uint16)
        self._last_legal_flags = np.zeros((self.num_envs, 4), dtype=bool)
        self._p4 = float(p4)

        # per-environment monotonic start time for rendering elapsed time
        # set when reset(...) is called for each env
        self._render_game_start_time = np.zeros((self.num_envs,), dtype=float)

        # 좌/우 LUT 준비 (최초 1회)
        if Batch2048Core._LUT_LEFT_NEW is None:
            lut_left_new, lut_right_new, lut_left_rew, lut_right_rew = (
                self._build_row_luts()
            )
            Batch2048Core._LUT_LEFT_NEW = lut_left_new
            Batch2048Core._LUT_RIGHT_NEW = lut_right_new
            Batch2048Core._LUT_LEFT_REW = lut_left_rew
            Batch2048Core._LUT_RIGHT_REW = lut_right_rew
            Batch2048Core._LUT_LR_NEW = np.stack(
                [
                    Batch2048Core._LUT_LEFT_NEW,
                    Batch2048Core._LUT_RIGHT_NEW,
                ],
                axis=1,
            )
            base = np.arange(65536, dtype=np.uint16)
            Batch2048Core._LUT_LEFT_MOV = Batch2048Core._LUT_LEFT_NEW != base
            Batch2048Core._LUT_RIGHT_MOV = Batch2048Core._LUT_RIGHT_NEW != base

        # 스폰 LUT 준비 (최초 1회)
        if Batch2048Core._LUT_EMPTY4_ROW is None:
            self._init_spawn_luts()

        self.render_mode = render_mode
        self._render_thread = None
        self._render_stop = threading.Event()
        if self.render_mode == "human":
            self._start_render_thread()

    def _start_render_thread(self):
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._render_thread.start()

    def _render_loop(self):
        pygame.init()
        # 보드/셀/여백 크기(Env2048 참고, 필요시 클래스 변수로 빼도 됨)
        size = 4
        tile_px = 100
        gap_px = 10
        grid_x = 20
        grid_y = 80
        grid_w = size * tile_px + (size + 1) * gap_px
        window_w = grid_w + 40
        window_h = grid_w + 120

        screen = pygame.display.set_mode((window_w, window_h))
        pygame.display.set_caption("Batch2048Core")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont("Arial", 24)
        font_big = pygame.font.SysFont("Arial", 36, bold=True)

        if self._render_game_start_time[0] == 0:
            self._render_game_start_time[0] = time.monotonic()

        try:
            # render-time tracking for max-tile updates
            # _render_max_tile: current best tile value shown
            # _render_max_tile_at: seconds-from-game-start when that tile was first observed
            self._render_max_tile = 0
            self._render_max_tile_at = None

            while not self._render_stop.is_set():
                # pump pygame events so window remains responsive
                for ev in pygame.event.get():
                    if ev.type == pygame.QUIT:
                        self._render_stop.set()
                        break

                screen.fill((250, 248, 239))
                # 제목/점수
                head_y = 20
                surf = font.render("Batch2048Core", True, FG_DARK)
                screen.blit(surf, (20, head_y))

                # 보드 영역 배경
                pygame.draw.rect(
                    screen,
                    BG_BOARD,
                    (grid_x, grid_y, grid_w, grid_w),
                    border_radius=10,
                )

                # 현재 보드 추출 (make a copy to avoid race with step())
                obs_snapshot = self._obs_func()
                obs_channel = obs_snapshot.shape[-1]
                if obs_channel == 4:
                    board_Nx4x4 = Batch2048Core.uint16x4_to_uint8x16(
                        obs_snapshot
                    ).reshape(-1, 4, 4)
                elif obs_channel == 16:
                    board_Nx4x4 = obs_snapshot.reshape(-1, 4, 4)
                elif obs_channel == 256:
                    grid = obs_snapshot.reshape(-1, 4, 4, 16)
                    board_Nx4x4 = np.argmax(grid, axis=-1)  # (N,4,4)
                else:
                    raise ValueError("Unsupported obs format in render_obs")

                board = board_Nx4x4[0]

                cell = tile_px
                gap = gap_px
                for r in range(4):
                    for c in range(4):
                        exp = int(board[r, c])
                        tile_val = 0 if exp <= 0 else (1 << exp)
                        color = TILE_COLORS.get(tile_val, (60, 58, 50))
                        x = grid_x + gap + c * (cell + gap)
                        y = grid_y + gap + r * (cell + gap)
                        pygame.draw.rect(
                            screen, color, (x, y, cell, cell), border_radius=8
                        )
                        if tile_val > 0:
                            fg = FG_DARK if tile_val <= 4 else FG_LIGHT
                            fsize = max(18, int(cell * 0.35))
                            font_dyn = pygame.font.SysFont("Arial", fsize, bold=True)
                            surf = font_dyn.render(str(tile_val), True, fg)
                            rect = surf.get_rect(center=(x + cell // 2, y + cell // 2))
                            screen.blit(surf, rect)

                # check max tile and update timestamp when it increases
                try:
                    current_max_exp = int(board.max())
                    current_max_val = (
                        0 if current_max_exp <= 0 else (1 << current_max_exp)
                    )
                except Exception:
                    current_max_val = 0

                # update max tile value; record the time-from-start when it first appeared
                if current_max_val > getattr(self, "_render_max_tile", 0):
                    self._render_max_tile = current_max_val
                    try:
                        start_t = float(self._render_game_start_time[0])
                        # store seconds from game start when this tile appeared
                        self._render_max_tile_at = max(0.0, time.monotonic() - start_t)
                    except Exception:
                        self._render_max_tile_at = None

                # render max-tile and the seconds-from-game-start when it updated (top-right)
                try:
                    if self._render_max_tile_at is None:
                        elapsed_display = "--"
                    else:
                        elapsed_display = f"{self._render_max_tile_at:.1f}s"
                    right_x = window_w - 20
                    top_y = 20
                    txt = f"Max: {self._render_max_tile}  ({elapsed_display})"
                    surf = font.render(txt, True, FG_DARK)
                    rect = surf.get_rect(topright=(right_x, top_y))
                    screen.blit(surf, rect)
                except Exception:
                    pass

                pygame.display.flip()
                # throttle to keep CPU low and keep window responsive
                # clock.tick(10)  # 10 FPS
        finally:
            pygame.quit()

    # ---------------- 공개 API ----------------

    def reset(
        self,
        *,
        seed: int | None = None,
        mask: np.ndarray | None = None,
        indices: np.ndarray | list | tuple | None = None,
    ):
        """
        Arg:
                seed: RNG 재시드용 시드값 (선택적)
                mask: (N,) bool 배열, True인 env만 리셋 (권장)
                indices: 리셋할 인덱스들 (레거시, mask 사용 권장)

        Returns:
                obs: obs_mode에 따른 관측 (N, ...)
                info: {"reset_mask": (N,) bool, "legal_actions": (N,4) bool}
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if (mask is not None) and (indices is not None):
            raise ValueError("Provide either mask or indices, not both.")

        # 1) mask 정규화
        if mask is None:
            if indices is None:
                # 전체 리셋
                reset_mask = np.ones((self.num_envs,), dtype=bool)
            else:
                idx = np.asarray(indices)
                if idx.dtype == np.bool_:
                    if idx.shape != (self.num_envs,):
                        raise ValueError(
                            f"indices(bool) shape must be ({self.num_envs},)"
                        )
                    reset_mask = idx.astype(bool, copy=False)
                else:
                    reset_mask = np.zeros((self.num_envs,), dtype=bool)
                    reset_mask[idx.astype(np.int64, copy=False)] = True
        else:
            if mask.dtype != np.bool_ or mask.shape != (self.num_envs,):
                raise ValueError(f"mask must be bool array of shape ({self.num_envs},)")
            reset_mask = mask

        # 2) 보드 초기화 + 스폰 (reset_mask=True인 곳만)
        if reset_mask.any():
            self._boards[reset_mask] = 0
            self._spawn_random_tile_batch_bitwise(
                self._boards, moved_mask=reset_mask, p4=self._p4
            )

        # 3) 합법 액션 플래그/정보
        canL, canR = self._compute_action_flags(self._boards)
        self._transpose_all(self._boards, out=self._boards_T)
        canU, canD = self._compute_action_flags(self._boards_T)
        legal_flags = np.stack([canL, canR, canU, canD], axis=1)

        info = {
            "reset_mask": reset_mask,  # (N,)
            "legal_actions": legal_flags,  # (N,4)
        }
        return self._obs_func(), info

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        Arg:
                actions: (N,) int64 배열, 값은 {0,1,2,3} (0=LEFT,1=RIGHT,2=UP,3=DOWN)

        Returns:
                obs: (N, ...) obs_mode에 따른 관측
                reward: (N,) float32 보상
                terminated: (N,) bool 합법 액션이 없을 때 True
                truncated: (N,) bool 항상 False (상위에서 time-limit 처리)
                info: dict{"invalid_move": (N,) bool, "legal_actions": (N,4) bool}
        """
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs,):
            raise ValueError(
                f"actions must have shape ({self.num_envs},), got {actions.shape}"
            )

        moved_mask = np.zeros((self.num_envs,), dtype=bool)
        reward_sum = np.zeros((self.num_envs,), dtype=np.int64)

        # 수평 액션 (LEFT/RIGHT)
        idx_h = np.nonzero((actions == 0) | (actions == 1))[0]
        if idx_h.size:
            idx_left = idx_h[actions[idx_h] == 0]
            if idx_left.size:
                moved, rew = self._apply_lut_inplace(
                    self._boards,
                    idx_left,
                    Batch2048Core._LUT_LEFT_NEW,
                    Batch2048Core._LUT_LEFT_MOV,
                    Batch2048Core._LUT_LEFT_REW,
                )
                moved_mask[idx_left] = moved
                reward_sum[idx_left] += rew

            idx_right = idx_h[actions[idx_h] == 1]
            if idx_right.size:
                moved, rew = self._apply_lut_inplace(
                    self._boards,
                    idx_right,
                    Batch2048Core._LUT_RIGHT_NEW,
                    Batch2048Core._LUT_RIGHT_MOV,
                    Batch2048Core._LUT_RIGHT_REW,
                )
                moved_mask[idx_right] = moved
                reward_sum[idx_right] += rew

        # 수직 액션 (UP/DOWN): 전치 보드에서 좌/우 LUT 적용
        self._transpose_all(self._boards, out=self._boards_T)
        idx_v = np.nonzero((actions == 2) | (actions == 3))[0]
        if idx_v.size:
            idx_up = idx_v[actions[idx_v] == 2]
            if idx_up.size:
                moved, rew = self._apply_lut_inplace(
                    self._boards_T,
                    idx_up,
                    Batch2048Core._LUT_LEFT_NEW,
                    Batch2048Core._LUT_LEFT_MOV,
                    Batch2048Core._LUT_LEFT_REW,
                )
                moved_mask[idx_up] = moved
                reward_sum[idx_up] += rew

            idx_down = idx_v[actions[idx_v] == 3]
            if idx_down.size:
                moved, rew = self._apply_lut_inplace(
                    self._boards_T,
                    idx_down,
                    Batch2048Core._LUT_RIGHT_NEW,
                    Batch2048Core._LUT_RIGHT_MOV,
                    Batch2048Core._LUT_RIGHT_REW,
                )
                moved_mask[idx_down] = moved
                reward_sum[idx_down] += rew

        # 이동된 보드에만 타일 스폰 (전치 상태에서 보드가 최신)
        self._spawn_random_tile_batch_bitwise(self._boards_T, moved_mask, p4=self._p4)

        # 다음 상태의 합법 액션 플래그 & 종료 판정
        canU, canD = self._compute_action_flags(self._boards_T)
        self._transpose_all(self._boards_T, out=self._boards)
        canL, canR = self._compute_action_flags(self._boards)
        legal_flags = np.stack([canL, canR, canU, canD], axis=1)
        terminated = ~legal_flags.any(axis=1)
        self._last_legal_flags = legal_flags

        obs = self._obs_func()
        reward = reward_sum.astype(np.float32, copy=False)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        info = {
            "invalid_move": ~moved_mask,  # 이번 액션에서 실제로 아무 변화도 없었는가
            "legal_actions": legal_flags,  # 다음 스텝에서 가능한 [L, R, U, D]
        }
        return obs, reward, terminated, truncated, info

    # ---------------- 관측 변환 ----------------
    @staticmethod
    def uint16x4_to_uint8x16(boards: np.ndarray) -> np.ndarray:
        """
        Arg:
                boards: (N,4) uint16 배열, 각 원소는 4칸의 지수값을 담은 16비트

        Returns:
                (N,16) uint8 배열, 각 칸의 지수 e 값
        """
        b = boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        return (
            np.stack([c0, c1, c2, c3], axis=2).astype(np.uint8).reshape(b.shape[0], 16)
        )

    def get_original_boards(self) -> np.ndarray:
        """
        Returns:
                내부 보드 상태 복사본 (N,4) uint16
        """
        return self._boards.copy()

    def boards_as_uint8x16(self) -> np.ndarray:
        """
        Returns:
                (N,16) uint8 배열, 각 칸의 지수 e 값
        """
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        return (
            np.stack([c0, c1, c2, c3], axis=2).astype(np.uint8).reshape(b.shape[0], 16)
        )

    def boards_onehot256(self, *, dtype=np.uint8, flatten: bool = True) -> np.ndarray:
        """
        Arg:
                dtype: 원-핫 인코딩에 사용할 데이터 타입 (기본값: np.uint8)
                flatten: True면 (N,256), False면 (N,16,16) 형태로 반환

        Returns:
                원-핫 인코딩된 배열 (N,256) 또는 (N,16,16)
        """
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        vals = np.stack([c0, c1, c2, c3], axis=2).reshape(b.shape[0], 16)  # (N,16)
        eye16 = np.eye(16, dtype=dtype)
        onehot = eye16[vals]  # (N,16,16)
        return onehot.reshape(b.shape[0], 16 * 16) if flatten else onehot

    # ---------------- 보조 메트릭 ----------------

    def best_tile(self) -> np.ndarray:
        """
        Returns:
                각 보드에서 가장 큰 타일의 지수값 (N,) uint8
        """
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        row_max = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3)).astype(np.uint8)
        return row_max.max(axis=1)

    def tile_score_sum(self) -> np.ndarray:
        """현재 보드의 타일 값 합 Σ 2^e (빈칸=0) → (N,)"""
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        v0 = np.where(c0 > 0, 1 << c0, 0)
        v1 = np.where(c1 > 0, 1 << c1, 0)
        v2 = np.where(c2 > 0, 1 << c2, 0)
        v3 = np.where(c3 > 0, 1 << c3, 0)
        return (v0 + v1 + v2 + v3).sum(axis=1)

    def estimated_cumulative_score(self, *, out_dtype=np.int64) -> np.ndarray:
        """
        Arg:
                out_dtype: 출력 데이터 타입 (기본값: np.int64)

        Returns:
                누적 점수 근사값 Σ_{e>0} 2^e * (e-1) (N,) 배열
                스폰이 항상 2였다고 가정 시 정확, 4가 섞이면 약간 과대추정
        """
        b = self._boards
        e0 = ((b >> 12) & 0xF).astype(np.int64)
        e1 = ((b >> 8) & 0xF).astype(np.int64)
        e2 = ((b >> 4) & 0xF).astype(np.int64)
        e3 = (b & 0xF).astype(np.int64)

        def contrib(e: np.ndarray) -> np.ndarray:
            return np.where(e > 0, (np.int64(1) << e) * (e - 1), 0)

        total = (contrib(e0) + contrib(e1) + contrib(e2) + contrib(e3)).sum(axis=1)
        return total.astype(out_dtype, copy=False)

    # ---------------- 내부 유틸/루틴 ----------------

    @staticmethod
    def _pack_row(vals: np.ndarray) -> int:
        # vals: (4,) uint8  [a b c d] (a가 상위니블)
        return (
            (int(vals[0]) << 12)
            | (int(vals[1]) << 8)
            | (int(vals[2]) << 4)
            | int(vals[3])
        )

    @staticmethod
    def _unpack_row(r: int) -> np.ndarray:
        return np.array(
            [(r >> 12) & 0xF, (r >> 8) & 0xF, (r >> 4) & 0xF, r & 0xF], dtype=np.uint8
        )

    @classmethod
    def _slide_merge_left_row(cls, vals: np.ndarray) -> np.uint16:
        # 보상 계산 없이 LEFT 결과 행만
        comp = [int(v) for v in vals if v != 0]
        out = []
        i = 0
        while i < len(comp):
            if i + 1 < len(comp) and comp[i] == comp[i + 1]:
                out.append(comp[i] + 1)
                i += 2
            else:
                out.append(comp[i])
                i += 1
        while len(out) < 4:
            out.append(0)
        return np.uint16(
            cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15))
        )

    @classmethod
    def _slide_merge_left_row_with_reward(
        cls, vals: np.ndarray
    ) -> tuple[np.uint16, int]:
        # LEFT로 슬라이드+머지: (새 행, 이번 이동 보상)
        comp = [int(v) for v in vals if v != 0]
        out = []
        i = 0
        reward = 0
        while i < len(comp):
            if i + 1 < len(comp) and comp[i] == comp[i + 1]:
                new_e = comp[i] + 1
                out.append(new_e)
                reward += 1 << new_e  # 2048 점수 룰
                i += 2
            else:
                out.append(comp[i])
                i += 1
        while len(out) < 4:
            out.append(0)
        row16 = np.uint16(
            cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15))
        )
        return row16, reward

    @classmethod
    def _build_row_luts(cls):
        """좌/우 결과행 및 보상 LUT 생성"""
        lut_left = np.zeros(65536, dtype=np.uint16)
        lut_right = np.zeros(65536, dtype=np.uint16)
        lut_left_rew = np.zeros(65536, dtype=np.uint32)
        lut_right_rew = np.zeros(65536, dtype=np.uint32)

        def reverse_row16(r: int) -> int:
            # abcd -> dcba
            return (
                ((r & 0x000F) << 12)
                | ((r & 0x00F0) << 4)
                | ((r & 0x0F00) >> 4)
                | ((r & 0xF000) >> 12)
            )

        for r in range(65536):
            orig = cls._unpack_row(r)
            left_r, left_rew = cls._slide_merge_left_row_with_reward(orig)
            lut_left[r] = left_r
            lut_left_rew[r] = left_rew

            rev = reverse_row16(r)
            rev_orig = cls._unpack_row(rev)
            rev_left_r, rev_left_rew = cls._slide_merge_left_row_with_reward(rev_orig)
            right_r = reverse_row16(int(rev_left_r))
            lut_right[r] = right_r
            lut_right_rew[r] = rev_left_rew

        return lut_left, lut_right, lut_left_rew, lut_right_rew

    @staticmethod
    def obs_shape(mode: ObsMode) -> tuple[int, ...]:
        if mode is Batch2048Core.ObsMode.UINT16x4:
            return (4,)
        if mode is Batch2048Core.ObsMode.UINT8x16:
            return (16,)
        if mode is Batch2048Core.ObsMode.ONEHOT256:
            return (256,)
        raise ValueError(mode)

    @staticmethod
    def obs_dtype(mode: ObsMode):
        if mode is Batch2048Core.ObsMode.UINT16x4:
            return np.uint16
        if mode is Batch2048Core.ObsMode.UINT8x16:
            return np.uint8
        if mode is Batch2048Core.ObsMode.ONEHOT256:
            return np.uint8
        raise ValueError(mode)

    def _select_obs_fn(self, mode: ObsMode):
        if mode is self.ObsMode.UINT16x4:
            return lambda: self._boards.copy()
        if mode is self.ObsMode.UINT8x16:
            return self.boards_as_uint8x16
        if mode is self.ObsMode.ONEHOT256:
            return self.boards_onehot256
        raise ValueError(f"Unsupported obs_mode: {mode}")

    def _apply_lut_inplace(
        self,
        target_board: np.ndarray,
        idx: np.ndarray,
        lut_rows: np.ndarray,
        lut_moved: np.ndarray,
        lut_rewards: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """선택된 보드에 좌/우 LUT 적용 및 보상 합산"""
        boards = target_board
        b0 = boards[idx, 0]
        b1 = boards[idx, 1]
        b2 = boards[idx, 2]
        b3 = boards[idx, 3]

        moved_any = lut_moved[b0] | lut_moved[b1] | lut_moved[b2] | lut_moved[b3]
        rewards = (
            lut_rewards[b0].astype(np.int64)
            + lut_rewards[b1].astype(np.int64)
            + lut_rewards[b2].astype(np.int64)
            + lut_rewards[b3].astype(np.int64)
        )

        boards[idx, 0] = lut_rows[b0]
        boards[idx, 1] = lut_rows[b1]
        boards[idx, 2] = lut_rows[b2]
        boards[idx, 3] = lut_rows[b3]

        return moved_any, rewards

    def _transpose_all(self, x: np.ndarray, out: np.ndarray):
        """(N,4) 행-니블 보드를 니블 전치하여 out에 저장"""
        a = x[:, 0]
        b = x[:, 1]
        c = x[:, 2]
        d = x[:, 3]
        t0 = (
            (a & 0xF000)
            | ((b & 0xF000) >> 4)
            | ((c & 0xF000) >> 8)
            | ((d & 0xF000) >> 12)
        )
        t1 = (
            ((a & 0x0F00) << 4)
            | (b & 0x0F00)
            | ((c & 0x0F00) >> 4)
            | ((d & 0x0F00) >> 8)
        )
        t2 = (
            ((a & 0x00F0) << 8)
            | ((b & 0x00F0) << 4)
            | (c & 0x00F0)
            | ((d & 0x00F0) >> 4)
        )
        t3 = (
            ((a & 0x000F) << 12)
            | ((b & 0x000F) << 8)
            | ((c & 0x000F) << 4)
            | (d & 0x000F)
        )
        out[:, 0] = t0
        out[:, 1] = t1
        out[:, 2] = t2
        out[:, 3] = t3

    def _compute_action_flags(
        self, target_board: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """현재 보드에서 (LEFT 가능?, RIGHT 가능?) 플래그 튜플 반환"""
        lut_L = Batch2048Core._LUT_LEFT_MOV
        lut_R = Batch2048Core._LUT_RIGHT_MOV
        b0 = target_board[:, 0]
        b1 = target_board[:, 1]
        b2 = target_board[:, 2]
        b3 = target_board[:, 3]
        canL = lut_L[b0] | lut_L[b1] | lut_L[b2] | lut_L[b3]
        canR = lut_R[b0] | lut_R[b1] | lut_R[b2] | lut_R[b3]
        return canL, canR

    def _init_spawn_luts(self):
        """스폰 최적화용 LUT 일괄 생성(최초 1회)"""
        pc4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)

        lut_sel4 = np.full((16, 4), 255, dtype=np.uint8)
        for mask in range(16):
            cols = []
            for col in range(4):  # col=0..3 (왼→오)
                bit = 3 - col  # bit3↔col0, bit0↔col3
                if (mask >> bit) & 1:
                    cols.append(col)
            for n, col in enumerate(cols):
                lut_sel4[mask, n] = col

        empty4 = np.zeros(65536, dtype=np.uint16)
        for r in range(65536):
            m3 = 1 if ((r & 0xF000) == 0) else 0
            m2 = 1 if ((r & 0x0F00) == 0) else 0
            m1 = 1 if ((r & 0x00F0) == 0) else 0
            m0 = 1 if ((r & 0x000F) == 0) else 0
            empty4[r] = (m3 << 3) | (m2 << 2) | (m1 << 1) | m0

        pc16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint16)

        lut_sel16_row = np.full((1 << 16, 16), 255, dtype=np.uint16)
        lut_sel16_col = np.full((1 << 16, 16), 255, dtype=np.uint16)
        for m in range(1 << 16):
            m0 = (m >> 12) & 0xF
            m1 = (m >> 8) & 0xF
            m2 = (m >> 4) & 0xF
            m3 = (m >> 0) & 0xF
            c0 = int(pc4[m0])
            c1 = int(pc4[m1])
            c2 = int(pc4[m2])
            c3 = int(pc4[m3])

            for n in range(c0):
                col = lut_sel4[m0, n]
                lut_sel16_row[m, n] = 0
                lut_sel16_col[m, n] = 3 - col
            base = c0
            for n in range(c1):
                col = lut_sel4[m1, n]
                lut_sel16_row[m, base + n] = 1
                lut_sel16_col[m, base + n] = 3 - col
            base += c1
            for n in range(c2):
                col = lut_sel4[m2, n]
                lut_sel16_row[m, base + n] = 2
                lut_sel16_col[m, base + n] = 3 - col
            base += c2
            for n in range(c3):
                col = lut_sel4[m3, n]
                lut_sel16_row[m, base + n] = 3
                lut_sel16_col[m, base + n] = 3 - col

        Batch2048Core._PC4 = pc4
        Batch2048Core._PC16 = pc16
        Batch2048Core._LUT_EMPTY4_ROW = empty4
        Batch2048Core._LUT_MASK_SELECT = lut_sel4
        Batch2048Core._LUT_SELECT16_ROWS = lut_sel16_row
        Batch2048Core._LUT_SELECT16_COLS_REVERSE = lut_sel16_col

    def _spawn_random_tile_batch_bitwise(
        self, target_board: np.ndarray, moved_mask: np.ndarray, p4: float = 0.1
    ):
        """
        이동이 있었던 보드들에만 타일 1개 스폰.
        - 빈칸 보드 마스크(16비트)를 LUT로 계산, 무작위 nth 빈칸을 골라 2/4 배치.
        """
        idx_env = np.nonzero(moved_mask)[0]
        if idx_env.size == 0:
            return

        empty4 = Batch2048Core._LUT_EMPTY4_ROW
        pc16 = Batch2048Core._PC16

        row_masks = empty4[target_board[idx_env]]  # (M,4)
        board_mask16 = (
            (row_masks[:, 0] << 12)
            | (row_masks[:, 1] << 8)
            | (row_masks[:, 2] << 4)
            | (row_masks[:, 3] << 0)
        )

        total_empty = pc16[board_mask16]  # (M,)
        valid = total_empty > 0
        if not np.any(valid):
            return

        env_ids = idx_env[valid]
        v_mask16 = board_mask16[valid]
        v_tot = total_empty[valid]

        rng = self._rng
        v_nth = rng.integers(0, v_tot, dtype=np.uint16)  # (Mv,)
        v_k = np.where(rng.random(size=v_tot.shape) < p4, 2, 1).astype(np.uint16)

        rows = Batch2048Core._LUT_SELECT16_ROWS[v_mask16, v_nth]
        cols = Batch2048Core._LUT_SELECT16_COLS_REVERSE[v_mask16, v_nth]
        shift = cols << 2  # 0,4,8,12

        target_board[env_ids, rows] |= v_k << shift

    # ---------------- 디버그용 ----------------

    # @staticmethod
    # def render_board_text(board_row16: np.ndarray) -> str:
    # 	"""
    # 	(4,) uint16 보드를 텍스트 그리드로 변환 (디버그 출력용)
    # 	각 칸은 실제 값(2^e)로 표시, 빈칸은 0.
    # 	"""
    # 	result = ""
    # 	for r in board_row16:
    # 		cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
    # 		result += " ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells) + "\n"
    # 	return result

    @staticmethod
    def render_obs(obs: np.ndarray) -> str:
        """
        obs_mode에 따른 관측을 텍스트 그리드로 변환 (디버그 출력용)
        각 칸은 실제 값(2^e)로 표시, 빈칸은 0.
        지원: (N,4)[uint16], (N,16)[uint8], (N,256)[uint8|float32]
        """
        obs_channel = obs.shape[-1]
        if obs.ndim != 2 or obs_channel not in (4, 16, 256):
            raise ValueError(
                "render_obs only supports obs with shape (N,4), (N,16), or (N,256)"
            )

        fmt = lambda e: f"{((1 << int(e)) if int(e) > 0 else 0):^5d}"
        lines = []

        if obs_channel == 4:
            board_Nx4x4 = Batch2048Core.uint16x4_to_uint8x16(obs).reshape(-1, 4, 4)
        elif obs_channel == 16:
            board_Nx4x4 = obs.reshape(-1, 4, 4)
        elif obs_channel == 256:
            grid = obs.reshape(-1, 4, 4, 16)
            exps = np.argmax(grid, axis=-1)  # (N,4,4)
            board_Nx4x4 = exps
        else:
            raise ValueError("Unsupported obs format in render_obs")

        for row in board_Nx4x4.swapaxes(0, 1):
            parts = [" ".join(fmt(e) for e in r) for r in row]
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    def close(self):
        self._render_stop.set()
        if self._render_thread is not None:
            self._render_thread.join()


def _single_obs_space_for(mode: Batch2048Core.ObsMode):
    shape = Batch2048Core.obs_shape(mode)
    dtype = Batch2048Core.obs_dtype(mode)
    # SB3는 단일 env 기준 space 필요
    if mode is Batch2048Core.ObsMode.UINT16x4:
        return spaces.Box(low=0, high=np.uint16(0xFFFF), shape=shape, dtype=dtype)
    if mode is Batch2048Core.ObsMode.UINT8x16:
        return spaces.Box(low=0, high=15, shape=shape, dtype=dtype)  # 지수 e 범위
    if mode is Batch2048Core.ObsMode.ONEHOT256:
        return spaces.Box(low=0, high=1, shape=shape, dtype=dtype)
    raise ValueError(mode)


# =========================
# 1) Gym 단일 환경 어댑터
# =========================


class Gym2048SingleEnv(gym.Env):
    """
    Gym 단일 환경 어댑터.
    - 내부적으로 Batch2048Core(num_envs=1)를 사용해 표준 Gym API를 제공.
    - 관측/행동 space는 단일 환경 기준으로 노출.
    - 시간 제한 같은 트렁케이션은 상위에서 래핑해 사용.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        obs_mode: Batch2048Core.ObsMode = Batch2048Core.ObsMode.UINT8x16,
        seed: int | None = None,
        p4: float = 0.1,
    ):
        super().__init__()
        self.obs_mode = obs_mode
        self.core = Batch2048Core(obs_mode=obs_mode, num_envs=1, seed=seed, p4=p4)

        # 단일 환경 기준 space
        self.observation_space = _single_obs_space_for(obs_mode)
        self.action_space = spaces.Discrete(4)  # 0=LEFT,1=RIGHT,2=UP,3=DOWN

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            obs, info = self.core.reset(seed=seed)
        else:
            obs, info = self.core.reset()
        # core는 (N, …) 반환 → 단일 env이므로 [0]
        return obs[0], self._slice_info(info, 0)

    def step(self, action):
        # action은 스칼라 → (1,)로 포장
        actions = np.array([int(action)], dtype=np.int64)
        obs, reward, terminated, truncated, info = self.core.step(actions)
        return (
            obs[0],
            float(reward[0]),
            bool(terminated[0]),
            bool(truncated[0]),
            self._slice_info(info, 0),
        )

    def render(self):
        # 텍스트 보드 렌더링
        row16 = self.core.get_original_boards()  # (N, 4) uint16
        return self.core.render_obs(row16)

    def _slice_info(self, info: dict, i: int) -> dict:
        # core의 info는 배치 형태의 항목을 가질 수 있음 → 단일 env로 잘라서 반환
        out = {}
        for k, v in info.items():
            if isinstance(v, np.ndarray) and v.shape[0] == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out


# =========================
# 2) SB3 VecEnv 어댑터
# =========================


class SB3Batch2048VecEnv(VecEnv):
    """
    Batch2048Core → SB3 호환 VecEnv 어댑터.
    - core.num_envs 개의 환경을 내부적으로 이미 벡터화하여 처리.
    - SB3가 요구하는 step_async/step_wait, 자동 부분 리셋, infos 리스트화 등을 수행.
    - observation_space / action_space는 '단일 환경 기준'으로 노출.
    """

    def __init__(self, core):
        # core는 Batch2048Core 인스턴스
        self.core = core
        self.num_envs = int(core.num_envs)

        # SB3의 VecEnv 계약: 단일 환경 기준 space를 노출
        self.observation_space = _single_obs_space_for(core.obs_mode)
        self.action_space = spaces.Discrete(4)

        # VecEnv 내부 상태
        self._waiting = False
        self._actions = None

    @classmethod
    def from_params(
        cls,
        *,
        obs_mode=Batch2048Core.ObsMode.UINT8x16,
        num_envs=1024,
        seed=None,
        p4=0.1,
    ):
        core = Batch2048Core(obs_mode=obs_mode, num_envs=num_envs, seed=seed, p4=p4)
        return cls(core)

    def reset(self):
        obs, info = self.core.reset()
        # SB3는 (num_envs, *obs_shape) 배열과 infos(list[dict])를 기대
        # infos = self._split_info(info)
        return obs

    def step_async(self, actions):
        """
        SB3가 넘겨주는 actions: shape=(num_envs,), dtype=int
        """
        if isinstance(actions, np.ndarray):
            if actions.shape != (self.num_envs,):
                raise ValueError(
                    f"actions must have shape ({self.num_envs},), got {actions.shape}"
                )
            self._actions = actions.astype(np.int64, copy=False)
        else:
            _arr = np.asarray(actions, dtype=np.int64)
            if _arr.shape != (self.num_envs,):
                raise ValueError(
                    f"actions must have shape ({self.num_envs},), got {_arr.shape}"
                )
            self._actions = _arr
        self._waiting = True

    def step_wait(self):
        assert self._waiting, "step_wait() called without step_async()"
        self._waiting = False

        obs_step, reward, terminated, truncated, info_step = self.core.step(
            self._actions
        )
        done = terminated | truncated

        if np.any(terminated):
            obs_before = obs_step.copy()
            obs_after, info_reset = self.core.reset(mask=terminated)
            infos = self._split_info(info_step)
            infos_reset = self._split_info(info_reset)
            for i in range(self.num_envs):
                # reset 관련 키(legal_actions/reset_mask 등) 추가
                infos[i].update(infos_reset[i])
            # terminal_observation 채우기(terminated인 env만)
            for i in np.nonzero(terminated)[0]:
                infos[i] = dict(infos[i])  # copy
                infos[i]["terminal_observation"] = obs_before[i]
            # 리턴할 obs는 리셋 반영된 최신 상태
            obs = obs_after
        else:
            obs = obs_step
            infos = self._split_info(info_step)

        return obs, reward, done, infos

    def close(self):
        pass

    # ---- VecEnv 필수 보조 메서드 ----

    def get_attr(self, name, indices=None):
        # 1) VecEnv 자신에게 있는 속성/메서드 우선
        if hasattr(self, name):
            return [getattr(self, name)]
        # 2) 그 다음 core에서 찾기
        if hasattr(self.core, name):
            return [getattr(self.core, name)]
        raise AttributeError(f"{name!r} not found on VecEnv or core")

    def set_attr(self, name, values, indices=None):
        setattr(self.core, name, values)

    def env_method(self, method_name, *args, indices=None, **kwargs):
        if hasattr(self, method_name):
            return [getattr(self, method_name)(*args, **kwargs)]
        if hasattr(self.core, method_name):
            return [getattr(self.core, method_name)(*args, **kwargs)]
        raise AttributeError(f"Method {method_name!r} not found on VecEnv or core")

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False]

    def action_masks(self) -> np.ndarray:
        # shape = (num_envs, n_actions) = (N, 4)
        # True = 유효, False = 불가
        return self.core._last_legal_flags.astype(np.bool_, copy=False)

    # ---- 내부 유틸 ----

    def _split_info(self, info_batch: dict) -> list[dict]:
        """
        core의 info(dict of batch-arrays)를 SB3가 기대하는 list[dict]로 변환.
        - 길이 num_envs의 리스트를 만들고, 각 키별로 i번째 항목만 슬라이스해서 dict 구성.
        """
        infos = [dict() for _ in range(self.num_envs)]
        for k, v in info_batch.items():
            # v가 배치 배열이면 env별로 슬라이스, 아니면 그대로 복제
            if isinstance(v, np.ndarray) and v.shape[0] == self.num_envs:
                for i in range(self.num_envs):
                    infos[i][k] = v[i]
            else:
                for i in range(self.num_envs):
                    infos[i][k] = v
        return infos


# if __name__ == "__main__":
# 	import tqdm  # 속도 체크용
# 	# 간단한 스모크 테스트 (SB3 없이 코어 단독 구동)
# 	core = Batch2048Core(obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=2**15, seed=42)
# 	obs, info = core.reset()
# 	print("Initial legal ratio:", info["legal_actions"].mean())
#
# 	total_steps = 1000 * (2**18 // core.num_envs)
# 	for _ in tqdm.tqdm(range(total_steps)):
# 		# 무작위 액션 (N,)
# 		actions = core._rng.integers(0, 4, size=core.num_envs, dtype=np.int64)
# 		obs, reward, terminated, truncated, info = core.step(actions)
#
# 		# 종료된 env만 부분 리셋
# 		if np.any(terminated):
# 			core.reset(mask=terminated)
#
# 	print("Mean best tile:", core.best_tile().mean())
# 	print("Mean tile sum:", core.tile_score_sum().mean())
# 	print("Mean score est.:", core.estimated_cumulative_score().mean())


if __name__ == "__main__":
    env = Batch2048Core(obs_mode=Batch2048Core.ObsMode.UINT8x16, num_envs=2, seed=42)
    obs, info = env.reset()
    print("Initial boards:")
    print(env.render_obs(obs))
    print("Info:", info)

    for step in range(5000):
        input_str = (
            input("Enter action (a=LEFT, d=RIGHT, w=UP, s=DOWN, q=quit): ")
            .strip()
            .lower()
        )
        if input_str == "q":
            break
        action_map = {"a": 0, "d": 1, "w": 2, "s": 3}
        if input_str not in action_map:
            print("Invalid input. Please enter a, d, w, s, or q.")
            continue
        actions = np.array([action_map[input_str]] * env.num_envs, dtype=np.int64)
        # actions = env._rng.integers(0, 4, size=env.num_envs, dtype=np.int64)  # 무작위 액션
        obs, reward, terminated, truncated, info = env.step(actions)
        obs, info2 = env.reset(mask=terminated)
        print(f"\nStep {step+1}, Actions: {actions}")
        print(
            f"Info_legal: {info['legal_actions']}, Info_invalid: {info['invalid_move']}"
        )
        print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        print("Boards:")
        print(env.render_obs(obs))
        print("Best tiles:", env.best_tile())
        print("Sum tiles:", env.tile_score_sum())
