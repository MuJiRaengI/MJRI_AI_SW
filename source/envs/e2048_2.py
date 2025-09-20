import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tqdm


class Batch2048EnvFast(gym.Env):
    """
    - 내부 상태: boards (N,4) uint16, 각 원소는 4칸 니블(상위→하위)
    - 내부에 원본 보드와 전치 보드를 모두 보관하여, 수평/수직 액션 시 한 번의 전치만 수행
    - 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    - 스폰/보상 미구현(요청대로). invalid_move만 info로 제공.
    """

    # 클래스 정적 LUT (좌/우 결과행) — __init__에서 필요 시 1회 생성
    _LUT_LEFT_NEW: np.ndarray | None = None  # uint16[65536]
    _LUT_RIGHT_NEW: np.ndarray | None = None  # uint16[65536]
    _LUT_LEFT_MOV: np.ndarray | None = None  # bool/uint8[65536]
    _LUT_RIGHT_MOV: np.ndarray | None = None  # bool/uint8[65536]
    _LUT_LR_NEW: np.ndarray | None = None  # uint16[65536,2] (0=left,1=right)

    # 스폰 최적화용 클래스 정적 LUT들
    _PC4: np.ndarray | None = None  # uint8[16], popcount
    _PC16: np.ndarray | None = None  # uint16[65536], popcount
    _LUT_EMPTY4_ROW: np.ndarray | None = None  # uint16[65536], 4비트 빈칸 마스크
    _LUT_MASK_SELECT: np.ndarray | None = (
        None  # uint8[16,4], (mask, nth)->col(0..3) or 255
    )
    _LUT_SELECT16_ROWS: np.ndarray | None = (
        None  # uint16[65536,16], (mask16,nth)->row or 255
    )
    _LUT_SELECT16_COLS_REVERSE: np.ndarray | None = (
        None  # uint16[65536,16], (mask16,nth)->col or 255
    )

    metadata = {"render_modes": []}

    def __init__(
        self,
        obs_mode: str,
        num_envs: int = 1024,
        seed: int | None = None,
        p4: float = 0.1,
    ):
        super().__init__()
        self.num_envs = int(num_envs)
        self.observation_space = spaces.Box(
            low=0, high=np.uint16(0xFFFF), shape=(self.num_envs, 4), dtype=np.uint16
        )
        self.action_space = spaces.MultiDiscrete([4] * self.num_envs)

        self._rng = np.random.default_rng(seed)
        self._boards = np.zeros((self.num_envs, 4), dtype=np.uint16)
        self._boards_T = np.zeros((self.num_envs, 4), dtype=np.uint16)
        self._last_legal_flags = np.zeros((self.num_envs, 4), dtype=bool)
        self._p4 = float(p4)

        self.obs_func = {
            "uint16x4": lambda: self._boards.copy(),
            "uint8x16": self.boards_as_uint8x16,
            "onehot256": self.boards_onehot256,
        }.get(obs_mode, None)

        # LUT(좌/우) 준비
        if Batch2048EnvFast._LUT_LEFT_NEW is None:
            Batch2048EnvFast._LUT_LEFT_NEW, Batch2048EnvFast._LUT_RIGHT_NEW = (
                self._build_row_luts()
            )
            Batch2048EnvFast._LUT_LR_NEW = np.stack(
                [
                    Batch2048EnvFast._LUT_LEFT_NEW,
                    Batch2048EnvFast._LUT_RIGHT_NEW,
                    Batch2048EnvFast._LUT_LEFT_NEW,
                    Batch2048EnvFast._LUT_RIGHT_NEW,
                ],
                axis=1,
            )
            base = np.arange(65536, dtype=np.uint16)
            Batch2048EnvFast._LUT_LEFT_MOV = Batch2048EnvFast._LUT_LEFT_NEW != base
            Batch2048EnvFast._LUT_RIGHT_MOV = Batch2048EnvFast._LUT_RIGHT_NEW != base

        # 스폰용 LUT들 준비 (최초 한 번)
        if Batch2048EnvFast._LUT_EMPTY4_ROW is None:
            self._init_spawn_luts()

    # ---------- 공개 API ----------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Normalize indices from options (support None, list/ndarray of ints, or boolean mask)
        if (
            options is not None
            and "indices" in options
            and options["indices"] is not None
        ):
            indices = options["indices"]
            # return self._boards.copy(), {}
            if isinstance(indices, (list, tuple)):
                idx = np.array(indices, dtype=np.int64)
            else:
                idx = np.asarray(indices)
                if idx.dtype == np.bool_:
                    if idx.shape != (self.num_envs,):
                        raise ValueError(
                            f"Boolean indices must have shape ({self.num_envs},), got {idx.shape}"
                        )
                    idx = np.nonzero(idx)[0].astype(np.int64)
                else:
                    idx = idx.astype(np.int64)

            if idx.size:
                self._boards[idx] = 0
                moved_mask = np.zeros((self.num_envs,), dtype=bool)
                moved_mask[idx] = True
                self._spawn_random_tile_batch_bitwise(self._boards, moved_mask, p4=0.1)
            info = {"reset_indices": idx}
        else:
            # Full reset (previous behavior)
            self._boards.fill(0)
            self._spawn_random_tile_batch_bitwise(
                self._boards,
                np.full((self.num_envs,), True),
                p4=self._p4,
            )
            info = {}

        # info["legal_actions"] =
        canL, canR = self._compute_action_flags(self._boards)
        self._transpose_all(self._boards, out=self._boards_T)
        canU, canD = self._compute_action_flags(self._boards_T)
        legal_flags = np.stack([canL, canR, canU, canD], axis=1)
        info["legal_actions"] = legal_flags

        obs = self.obs_func()
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        action: (N,) int64 in {0,1,2,3}
        - 수평 액션(0/1)인데 현재 전치 상태면 → 전치 해제(원상태로)
        - 수직 액션(2/3)인데 현재 비전치면 → 전치 적용(수평화)
        그 뒤, 좌/우 LUT로 행별 변환. 복원 없이 전치 플래그만 유지/토글.
        """
        # invalid = np.zeros((self.num_envs,), dtype=bool)
        moved_mask = np.zeros((self.num_envs,), dtype=bool)

        # Horizontal
        idx_h = np.nonzero((action == 0) | (action == 1))[0]
        if idx_h.size:
            # Horizontal: LEFT
            idx_left = idx_h[action[idx_h] == 0]
            if idx_left.size:
                moved_mask[idx_left] = self._apply_lut_inplace(
                    self._boards,
                    idx_left,
                    Batch2048EnvFast._LUT_LEFT_NEW,
                    Batch2048EnvFast._LUT_LEFT_MOV,
                )

            # Horizontal: RIGHT
            idx_right = idx_h[action[idx_h] == 1]
            if idx_right.size:
                moved_mask[idx_right] = self._apply_lut_inplace(
                    self._boards,
                    idx_right,
                    Batch2048EnvFast._LUT_RIGHT_NEW,
                    Batch2048EnvFast._LUT_RIGHT_MOV,
                )

        self._transpose_all(self._boards, out=self._boards_T)

        # Vertical: transpose, apply, transpose-back
        idx_v = np.nonzero((action == 2) | (action == 3))[0]
        if idx_v.size:
            # UP -> LEFT while transposed
            idx_up = idx_v[action[idx_v] == 2]
            if idx_up.size:
                moved_mask[idx_up] = self._apply_lut_inplace(
                    self._boards_T,
                    idx_up,
                    Batch2048EnvFast._LUT_LEFT_NEW,
                    Batch2048EnvFast._LUT_LEFT_MOV,
                )
            # DOWN -> RIGHT while transposed
            idx_down = idx_v[action[idx_v] == 3]
            if idx_down.size:
                moved_mask[idx_down] = self._apply_lut_inplace(
                    self._boards_T,
                    idx_down,
                    Batch2048EnvFast._LUT_RIGHT_NEW,
                    Batch2048EnvFast._LUT_RIGHT_MOV,
                )

        # 3) 이동된 보드에 대해 타일 스폰
        self._spawn_random_tile_batch_bitwise(self._boards_T, moved_mask, p4=self._p4)

        # 4) 종료/합법 액션 플래그 계산 (스폰 이후 보드 기준, 전체 배치 벡터화)
        canU, canD = self._compute_action_flags(self._boards_T)
        self._transpose_all(self._boards_T, out=self._boards)
        canL, canR = self._compute_action_flags(self._boards)
        legal_flags = np.stack([canL, canR, canU, canD], axis=1)
        terminated = ~legal_flags.any(axis=1)
        self._last_legal_flags = legal_flags

        obs = self.obs_func()
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        truncated = np.zeros((self.num_envs,), dtype=np.bool_)
        info = {
            "invalid_move": ~moved_mask,  # 이번 액션이 무효였는지
            "legal_actions": legal_flags,  # 다음 스텝에서 가능한 [L, R, U, D]
        }
        return obs, reward, terminated, truncated, info

    def boards_as_uint8x16(self) -> np.ndarray:
        """
        현재 보드를 (N,4) uint16 -> (N,16) uint8 로 변환.
        각 칸은 4비트 값(=2의 지수)이 uint8로 확장된다.
        """
        b = self._boards  # (N,4) uint16
        # 4니블을 각각 추출
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        # (N,4,4) -> (N,16)
        expanded = (
            np.stack([c0, c1, c2, c3], axis=2).astype(np.uint8).reshape(b.shape[0], 16)
        )
        return expanded

    def boards_onehot256(self, *, dtype=np.uint8, flatten: bool = True) -> np.ndarray:
        """
        보드를 one-hot(니블 0..15)로 인코딩.
        - 반환 shape:
          - flatten=True  -> (N, 256)   = 16칸 × 16클래스
          - flatten=False -> (N, 16, 16) = (칸 16, 클래스 16)
        - dtype: np.uint8(0/1) 기본, 학습용이면 np.float32 권장
        """
        # (N,16) 니블값(0..15)으로 전개
        b = self._boards  # (N,4) uint16
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        vals = np.stack([c0, c1, c2, c3], axis=2).reshape(b.shape[0], 16)  # (N,16)

        # one-hot: (N,16,16) = (칸, 클래스)
        eye16 = np.eye(16, dtype=dtype)  # (16,16)
        onehot = eye16[vals]  # (N,16,16)

        if flatten:
            return onehot.reshape(b.shape[0], 16 * 16)  # (N,256)
        return onehot  # (N,16,16)

    def best_tile(self) -> np.ndarray:
        """
        각 보드에서 가장 큰 타일(4비트 값)을 반환 (N,) uint8
        """
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF
        row_max = np.maximum(np.maximum(c0, c1), np.maximum(c2, c3)).astype(np.uint8)
        return row_max.max(axis=1)

    def tile_score_sum(self) -> np.ndarray:
        b = self._boards
        c0 = (b >> 12) & 0xF
        c1 = (b >> 8) & 0xF
        c2 = (b >> 4) & 0xF
        c3 = b & 0xF

        # 각 니블을 실제 점수(2^x)로 변환 (빈칸=0)
        v0 = np.where(c0 > 0, 1 << c0, 0)
        v1 = np.where(c1 > 0, 1 << c1, 0)
        v2 = np.where(c2 > 0, 1 << c2, 0)
        v3 = np.where(c3 > 0, 1 << c3, 0)

        row_sum = v0 + v1 + v2 + v3  # (N,4)
        return row_sum.sum(axis=1)

    def estimated_cumulative_score(self, *, out_dtype=np.int64) -> np.ndarray:
        """
        현재 보드에서 누적 점수를 한 번에 근사 계산.
        공식: 각 칸의 지수 e>0 에 대해 2^e * (e-1) 를 합산
        - 반환: (N,) out_dtype
        - 스폰이 전부 2였다고 가정하면 정확, 4 스폰이 섞이면 약간 과대추정될 수 있음.
        """
        b = self._boards  # (N,4) uint16

        # 니블 추출 (각 칸의 지수 e: 0..15)
        e0 = ((b >> 12) & 0xF).astype(np.int64)
        e1 = ((b >> 8) & 0xF).astype(np.int64)
        e2 = ((b >> 4) & 0xF).astype(np.int64)
        e3 = (b & 0xF).astype(np.int64)

        def contrib(e: np.ndarray) -> np.ndarray:
            # e>0 에 대해서만 2^e * (e-1), e=0이면 0
            return np.where(e > 0, (np.int64(1) << e) * (e - 1), 0)

        # (N,4) 합 → (N,)
        total = (contrib(e0) + contrib(e1) + contrib(e2) + contrib(e3)).sum(axis=1)
        return total.astype(out_dtype, copy=False)

    # ---------- 유틸 (모두 클래스/인스턴스 메소드, 전역 없음) ----------

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
        # 보상/합쳐짐 여부는 무시하고 결과 행만 반환 (LUT 용)
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
    def _build_row_luts(cls):
        """
        좌/우 결과행 LUT 생성(보상/플래그 없음).
        - LEFT:  r -> left(r)
        - RIGHT: r -> reverse(left(reverse(r)))  (빌드 타임에 계산해 런타임 reverse 제거)
        """
        lut_left = np.zeros(65536, dtype=np.uint16)
        lut_right = np.zeros(65536, dtype=np.uint16)

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
            # LEFT
            left_r = cls._slide_merge_left_row(orig)
            lut_left[r] = left_r

            # RIGHT (빌드 타임에 역방향 LUT 고정)
            rev = reverse_row16(r)
            rev_orig = cls._unpack_row(rev)
            rev_left = cls._slide_merge_left_row(rev_orig)
            right_r = reverse_row16(int(rev_left))
            lut_right[r] = right_r

        return lut_left, lut_right

    def _apply_lut_inplace(
        self,
        target_board: np.ndarray,
        idx: np.ndarray,
        lut_rows: np.ndarray,
        lut_moved: np.ndarray,
    ) -> np.ndarray:
        """
        선택된 보드 idx에 대해 좌/우 LUT를 적용하고,
        LUT 기반의 '행 변화 여부'를 통해 per-board moved(boolean)를 반환한다.

        고속화를 위한 구현 디테일:
        - RHS 접근시 중간 (M,4) 복사본 생성을 피하기 위해 열 단위로 나눠 처리한다.
        - lut_rows: uint16[65536], 한 행(row16)에 대한 변환 결과
        - lut_moved: bool[65536], 해당 행이 변했는지 여부
        반환값:
        - moved_any: (len(idx),) bool, 각 보드에 대해 최소 1개 행이라도 변했는지
        """
        boards = target_board
        board_0 = boards[idx, 0]
        board_1 = boards[idx, 1]
        board_2 = boards[idx, 2]
        board_3 = boards[idx, 3]

        # 1) 변화 여부: LUT에서 각 열의 '변화 여부' 조회 후 OR
        moved_any = (
            lut_moved[board_0]
            | lut_moved[board_1]
            | lut_moved[board_2]
            | lut_moved[board_3]
        )

        # 2) LUT 결과로 실제 보드 값을 in-place 갱신
        boards[idx, 0] = lut_rows[board_0]
        boards[idx, 1] = lut_rows[board_1]
        boards[idx, 2] = lut_rows[board_2]
        boards[idx, 3] = lut_rows[board_3]

        return moved_any

    def _transpose_inplace(self, idx: np.ndarray):
        """
        비트연산 전치 (4x4 니블).
        boards[idx]: (M,4) uint16 의 각 보드를 전치하여 다시 (M,4)에 저장.
        """
        sub = self._boards[idx]
        a = sub[:, 0]
        b = sub[:, 1]
        c = sub[:, 2]
        d = sub[:, 3]

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

        self._boards[idx, 0] = t0
        self._boards[idx, 1] = t1
        self._boards[idx, 2] = t2
        self._boards[idx, 3] = t3

    def _transpose_inplace_all(self):
        """
        비트연산 전치 (4x4 니블).
        self._boards: (N,4) uint16 의 각 보드를 전치하여 self._boards_T에 저장.
        """
        a = self._boards[:, 0]
        b = self._boards[:, 1]
        c = self._boards[:, 2]
        d = self._boards[:, 3]

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

        self._boards[:, 0] = t0
        self._boards[:, 1] = t1
        self._boards[:, 2] = t2
        self._boards[:, 3] = t3

    def _transpose_all(self, x, out: np.ndarray):
        """
        Transpose all boards from x (N,4) to out (N,4) using nibble ops.
        """
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

    def _compute_action_flags(self, target_board: np.ndarray):
        lut_L = Batch2048EnvFast._LUT_LEFT_MOV
        lut_R = Batch2048EnvFast._LUT_RIGHT_MOV
        board_0 = target_board[:, 0]
        board_1 = target_board[:, 1]
        board_2 = target_board[:, 2]
        board_3 = target_board[:, 3]
        a = lut_L[board_0] | lut_L[board_1] | lut_L[board_2] | lut_L[board_3]
        b = lut_R[board_0] | lut_R[board_1] | lut_R[board_2] | lut_R[board_3]
        return a, b

    def _init_spawn_luts(self):
        """
        스폰 최적화용 LUT를 한 번만 생성.
        - _LUT_EMPTY4_ROW[row16]: 4비트 마스크 (bit3=col0, ..., bit0=col3), 해당 니블==0이면 1
        - _PC4[v]: v(0..15)에서 1의 개수
        - _PC16[v]: v(0..65535)에서 1의 개수 (보드 마스크용)
        - _LUT_MASK_SELECT[mask4, n]: mask4에서 n번째(0-index) 1비트의 열 인덱스(0..3), 없으면 255
        - _LUT_SELECT16[mask16, nth] -> (row, col) (없으면 255,255)
        """
        # 1) 4비트 popcount
        pc4 = np.array([bin(i).count("1") for i in range(16)], dtype=np.uint8)

        # 2) mask4 + nth -> col (O(1) 선택)
        lut_sel4 = np.full((16, 4), 255, dtype=np.uint8)
        for mask in range(16):
            cols = []
            for col in range(4):  # col=0..3 (왼→오)
                bit = 3 - col  # bit3↔col0, bit0↔col3
                if (mask >> bit) & 1:
                    cols.append(col)
            for n, col in enumerate(cols):
                lut_sel4[mask, n] = col

        # 3) row16 -> empty 4bit mask
        empty4 = np.zeros(65536, dtype=np.uint16)  # uint16으로 변경
        for r in range(65536):
            m3 = 1 if ((r & 0xF000) == 0) else 0
            m2 = 1 if ((r & 0x0F00) == 0) else 0
            m1 = 1 if ((r & 0x00F0) == 0) else 0
            m0 = 1 if ((r & 0x000F) == 0) else 0
            empty4[r] = (m3 << 3) | (m2 << 2) | (m1 << 1) | m0

        # 4) 16비트 popcount (보드 마스크용)
        pc16 = np.array(
            [bin(i).count("1") for i in range(1 << 16)], dtype=np.uint16
        )  # uint16으로 변경

        # 5) mask16 + nth -> (row, col)
        # mask16은 [row0(상위 4비트), row1, row2, row3(하위 4비트)]를 이어붙인 16비트
        lut_sel16_row = np.full((1 << 16, 16), 255, dtype=np.uint16)  # uint16으로 변경
        lut_sel16_col = np.full((1 << 16, 16), 255, dtype=np.uint16)  # uint16으로 변경
        for m in range(1 << 16):
            # 각 행의 4bit 추출
            m0 = (m >> 12) & 0xF  # row0
            m1 = (m >> 8) & 0xF  # row1
            m2 = (m >> 4) & 0xF  # row2
            m3 = (m >> 0) & 0xF  # row3
            c0 = int(pc4[m0])
            c1 = int(pc4[m1])
            c2 = int(pc4[m2])
            c3 = int(pc4[m3])

            # row0에서 가능한 n
            for n in range(c0):
                col = lut_sel4[m0, n]
                lut_sel16_row[m, n] = 0
                lut_sel16_col[m, n] = 3 - col
            # row1
            base = c0
            for n in range(c1):
                col = lut_sel4[m1, n]
                lut_sel16_row[m, base + n] = 1
                lut_sel16_col[m, base + n] = 3 - col
            # row2
            base += c1
            for n in range(c2):
                col = lut_sel4[m2, n]
                lut_sel16_row[m, base + n] = 2
                lut_sel16_col[m, base + n] = 3 - col
            # row3
            base += c2
            for n in range(c3):
                col = lut_sel4[m3, n]
                lut_sel16_row[m, base + n] = 3
                lut_sel16_col[m, base + n] = 3 - col
        # base+c3 == pc16[m]개의 유효 엔트리만 채워짐 (나머지는 255)

        Batch2048EnvFast._PC4 = pc4
        Batch2048EnvFast._PC16 = pc16
        Batch2048EnvFast._LUT_EMPTY4_ROW = empty4
        Batch2048EnvFast._LUT_MASK_SELECT = lut_sel4
        Batch2048EnvFast._LUT_SELECT16_ROWS = lut_sel16_row
        Batch2048EnvFast._LUT_SELECT16_COLS_REVERSE = lut_sel16_col

    def _spawn_random_tile_batch_bitwise(
        self,
        target_board: np.ndarray,
        moved_mask: np.ndarray,
        p4: float = 0.1,
    ):
        """
        보드 단위 16비트 빈칸 플래그(LUT)로 완전 벡터 스폰.
        - moved_mask: (N,) bool
        """
        idx_env = np.nonzero(moved_mask)[0]
        if idx_env.size == 0:
            return

        empty4 = Batch2048EnvFast._LUT_EMPTY4_ROW  # uint8[65536]
        pc16 = Batch2048EnvFast._PC16  # uint8[65536]

        # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
        row_masks = empty4[target_board[idx_env]]  # (M,4)
        board_mask16 = (
            (row_masks[:, 0] << 12)
            | (row_masks[:, 1] << 8)
            | (row_masks[:, 2] << 4)
            | (row_masks[:, 3] << 0)
        )  # (M,)

        # 2) 총 빈칸 수 & 유효 보드
        total_empty = pc16[board_mask16]  # (M,)
        valid = total_empty > 0
        if not np.any(valid):
            return

        env_ids = idx_env[valid]  # (Mv,)
        v_mask16 = board_mask16[valid]
        v_tot = total_empty[valid]

        # 3) nth & k 샘플
        rng = self._rng
        v_nth = rng.integers(0, v_tot, dtype=np.uint16)  # (Mv,)
        v_k = np.where(rng.random(size=v_tot.shape) < p4, 2, 1).astype(np.uint16)

        # 4) (row, col) = LUT16[mask16, nth]
        rows = Batch2048EnvFast._LUT_SELECT16_ROWS[v_mask16, v_nth]  # (Mv,)
        cols = Batch2048EnvFast._LUT_SELECT16_COLS_REVERSE[v_mask16, v_nth]  # (Mv,)

        # 5) 니블 세팅 (scatter) — base buffer 직접 갱신
        shift = cols << 2  # 0,4,8,12
        target_board[env_ids, rows] |= v_k << shift

    def _render_board(self, board: np.ndarray):
        # board: (4,) uint16
        result = ""
        for r in board:
            cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
            result += " ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells) + "\n"
        return result


if __name__ == "__main__":
    import sys

    env = Batch2048EnvFast(obs_mode="uint8x16", num_envs=2**15, seed=42)
    obs, info = env.reset()
    print("Initial boards:")
    print(obs)
    print("Info:", info)

    for step in tqdm.tqdm(range(1000 * 2**18 // env.num_envs), file=sys.stdout):
        actions = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(actions)
        if np.any(terminated):  # 종료된 환경 자동 리셋
            obs, info2 = env.reset(options={"indices": np.nonzero(terminated)[0]})
        # if terminated.all():
        # 	# print(f"All terminated at step {step+1}")
        # 	# #평균 점수
        # 	# print("Best tiles (max 15=32768):", env.best_tile().mean())
        # 	# print("Sum tiles:", env.tile_score_sum().mean())
        # 	# print("Estimated cumulative score:", env.estimated_cumulative_score().mean())
        # 	obs, info2 = env.reset()


# if __name__ == "__main__":
# 	def print_board(obs: np.ndarray):
# 		# obs: (N, 4, 4) uint8
# 		for row in obs.reshape(-1, 4, 4).swapaxes(0, 1):
# 			for r in row:
# 				print(" ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in r), end="   |   ")
# 			print()
# 	env = Batch2048EnvFast(obs_mode='uint8x16', num_envs=1, seed=42)
# 	obs, info = env.reset()
# 	print("Initial boards:")
# 	print_board(obs)
# 	print("Info:", info)
#
# 	for step in range(1000):
# 		input_str = input("Enter action (a=LEFT, d=RIGHT, w=UP, s=DOWN, q=quit): ").strip().lower()
# 		if input_str == 'q':
# 			break
# 		action_map = {'a': 0, 'd': 1, 'w': 2, 's': 3}
# 		if input_str not in action_map:
# 			print("Invalid input. Please enter a, d, w, s, or q.")
# 			continue
# 		actions = np.array([action_map[input_str]] * env.num_envs, dtype=np.int64)
# 		# actions = env.action_space.sample()
# 		obs, reward, terminated, truncated, info = env.step(actions)
# 		obs, info2 = env.reset(options={"indices": np.nonzero(terminated)[0]})
# 		print(f"\nStep {step+1}, Actions: {actions}")
# 		print(f"Info_legal: {info['legal_actions']}, Info_invalid: {info['invalid_move']}")
# 		print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
# 		print("Boards:")
# 		print_board(obs)
# 		print("Best tiles:", env.best_tile())
# 		print("Sum tiles:", env.tile_score_sum())
