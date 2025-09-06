# tabs-only indentation

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tqdm


class Batch2048EnvFast(gym.Env):
    """
    - 내부 상태: boards (N,4) uint16, 각 원소는 4칸 니블(상위→하위)
    - 전치 상태: is_transposed (N,) bool
    - 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    - 스폰/보상 미구현(요청대로). invalid_move만 info로 제공.
    """

    # 클래스 정적 LUT (좌/우 결과행) — __init__에서 필요 시 1회 생성
    _LUT_LEFT_NEW: np.ndarray | None = None  # uint16[65536]
    _LUT_RIGHT_NEW: np.ndarray | None = None  # uint16[65536]

    # 스폰 최적화용 클래스 정적 LUT들
    _PC4: np.ndarray | None = None  # uint8[16], popcount
    _PC16: np.ndarray | None = None  # uint8[65536], popcount
    _LUT_EMPTY4_ROW: np.ndarray | None = None  # uint8[65536], 4비트 빈칸 마스크
    _LUT_MASK_SELECT: np.ndarray | None = (
        None  # uint8[16,4], (mask, nth)->col(0..3) or 255
    )
    _LUT_SELECT16: np.ndarray | None = (
        None  # uint8[65536,16,2], (mask16,nth)->(row,col) or (255,255)
    )

    metadata = {"render_modes": []}

    def __init__(self, num_envs: int = 1024, seed: int | None = None):
        super().__init__()
        self.num_envs = int(num_envs)
        self.observation_space = spaces.Box(
            low=0, high=np.uint16(0xFFFF), shape=(self.num_envs, 4), dtype=np.uint16
        )
        self.action_space = spaces.MultiDiscrete([4] * self.num_envs)

        self._rng = np.random.default_rng(seed)
        self._boards = np.zeros((self.num_envs, 4), dtype=np.uint16)
        # Removed self._is_transposed

        # LUT(좌/우) 준비
        if Batch2048EnvFast._LUT_LEFT_NEW is None:
            Batch2048EnvFast._LUT_LEFT_NEW, Batch2048EnvFast._LUT_RIGHT_NEW = (
                self._build_row_luts()
            )

        # 스폰용 LUT들 준비 (최초 한 번)
        if Batch2048EnvFast._LUT_EMPTY4_ROW is None:
            self._init_spawn_luts()

    # ---------- 공개 API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._boards.fill(0)
        # Removed self._is_transposed.fill(False)
        self._spawn_random_tile_batch_bitwise(np.full((self.num_envs,), True), p4=0.1)
        obs = self._boards.copy()
        info = {}
        return obs, info

    def step(self, action: int | list | tuple | np.ndarray):
        """
        action: (N,) int64 in {0,1,2,3}
        - 수평 액션(0/1)인데 현재 전치 상태면 → 전치 해제(원상태로)
        - 수직 액션(2/3)인데 현재 비전치면 → 전치 적용(수평화)
        그 뒤, 좌/우 LUT로 행별 변환. 복원 없이 전치 플래그만 유지/토글.
        """
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = action.astype(np.int64)
        else:
            action = np.full((self.num_envs,), int(action), dtype=np.int64)

        assert action.shape == (self.num_envs,)

        # Removed transpose state alignment block using _is_transposed

        invalid = np.zeros((self.num_envs,), dtype=bool)

        # Horizontal: LEFT
        idx = np.nonzero(action == 0)[0]
        if idx.size:
            before = self._boards[idx].copy()
            self._apply_lut_inplace(idx, Batch2048EnvFast._LUT_LEFT_NEW)
            invalid[idx] = (self._boards[idx] == before).all(axis=1)

        # Horizontal: RIGHT
        idx = np.nonzero(action == 1)[0]
        if idx.size:
            before = self._boards[idx].copy()
            self._apply_lut_inplace(idx, Batch2048EnvFast._LUT_RIGHT_NEW)
            invalid[idx] = (self._boards[idx] == before).all(axis=1)

        # Vertical: transpose, apply, transpose-back
        idx_v = np.nonzero((action == 2) | (action == 3))[0]
        if idx_v.size:
            # transpose selected boards
            self._transpose_inplace(idx_v)
            # UP -> LEFT while transposed
            idx_up = idx_v[action[idx_v] == 2]
            if idx_up.size:
                before = self._boards[idx_up].copy()
                self._apply_lut_inplace(idx_up, Batch2048EnvFast._LUT_LEFT_NEW)
                invalid[idx_up] = (self._boards[idx_up] == before).all(axis=1)
            # DOWN -> RIGHT while transposed
            idx_down = idx_v[action[idx_v] == 3]
            if idx_down.size:
                before = self._boards[idx_down].copy()
                self._apply_lut_inplace(idx_down, Batch2048EnvFast._LUT_RIGHT_NEW)
                invalid[idx_down] = (self._boards[idx_down] == before).all(axis=1)
            # transpose back
            self._transpose_inplace(idx_v)

        # 3) 이동된 보드에 대해 타일 스폰
        moved_mask = ~invalid
        self._spawn_random_tile_batch_bitwise(moved_mask, p4=0.1)

        # 4) 종료/스폰은 보류(요청). reward=0, done=False
        obs = self._boards.copy()
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        terminated = np.zeros((self.num_envs,), dtype=bool)
        truncated = np.zeros((self.num_envs,), dtype=bool)
        info = {
            "invalid_move": invalid,  # 이번 액션이 무효였는지
        }
        return obs, reward, terminated, truncated, info

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

    def _apply_lut_inplace(self, idx: np.ndarray, lut_rows: np.ndarray):
        """
        선택된 보드 idx의 4개 행에 대해 주어진 수평 LUT를 적용.
        행마다 독립적으로 1D 인덱싱 → 벡터화.
        """
        sub = self._boards[idx]  # (M,4)
        # 각 행에 대해 LUT 인덱싱
        sub[:, 0] = lut_rows[sub[:, 0]]
        sub[:, 1] = lut_rows[sub[:, 1]]
        sub[:, 2] = lut_rows[sub[:, 2]]
        sub[:, 3] = lut_rows[sub[:, 3]]
        self._boards[idx] = sub

    def _transpose_inplace(self, idx: np.ndarray):
        """
        비트연산 전치 (4x4 니블).
        boards[idx]: (M,4) uint16 의 각 보드를 전치하여 다시 (M,4)에 저장.
        """
        sub = self._boards[idx]  # (M,4), 각 행은 0xABCD 니블들
        a = sub[:, 0]
        b = sub[:, 1]
        c = sub[:, 2]
        d = sub[:, 3]

        # t0 = [a0 b0 c0 d0] (각 행의 상위 니블들을 12,8,4,0 위치로)
        t0 = (
            (a & 0xF000)
            | ((b & 0xF000) >> 4)
            | ((c & 0xF000) >> 8)
            | ((d & 0xF000) >> 12)
        )
        # t1 = [a1 b1 c1 d1] (0x0F00)
        t1 = (
            ((a & 0x0F00) << 4)
            | (b & 0x0F00)
            | ((c & 0x0F00) >> 4)
            | ((d & 0x0F00) >> 8)
        )
        # t2 = [a2 b2 c2 d2] (0x00F0)
        t2 = (
            ((a & 0x00F0) << 8)
            | ((b & 0x00F0) << 4)
            | (c & 0x00F0)
            | ((d & 0x00F0) >> 4)
        )
        # t3 = [a3 b3 c3 d3] (0x000F)
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
        empty4 = np.zeros(65536, dtype=np.uint8)
        for r in range(65536):
            m3 = 1 if ((r & 0xF000) == 0) else 0
            m2 = 1 if ((r & 0x0F00) == 0) else 0
            m1 = 1 if ((r & 0x00F0) == 0) else 0
            m0 = 1 if ((r & 0x000F) == 0) else 0
            empty4[r] = (m3 << 3) | (m2 << 2) | (m1 << 1) | m0

        # 4) 16비트 popcount (보드 마스크용)
        pc16 = np.array([bin(i).count("1") for i in range(1 << 16)], dtype=np.uint8)

        # 5) mask16 + nth -> (row, col)
        # mask16은 [row0(상위 4비트), row1, row2, row3(하위 4비트)]를 이어붙인 16비트
        lut_sel16 = np.full(
            (1 << 16, 16, 2), 255, dtype=np.uint8
        )  # (row,col) = (255,255) = invalid
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
                lut_sel16[m, n, 0] = 0
                lut_sel16[m, n, 1] = col
            # row1
            base = c0
            for n in range(c1):
                col = lut_sel4[m1, n]
                lut_sel16[m, base + n, 0] = 1
                lut_sel16[m, base + n, 1] = col
            # row2
            base += c1
            for n in range(c2):
                col = lut_sel4[m2, n]
                lut_sel16[m, base + n, 0] = 2
                lut_sel16[m, base + n, 1] = col
            # row3
            base += c2
            for n in range(c3):
                col = lut_sel4[m3, n]
                lut_sel16[m, base + n, 0] = 3
                lut_sel16[m, base + n, 1] = col
        # base+c3 == pc16[m]개의 유효 엔트리만 채워짐 (나머지는 255)

        Batch2048EnvFast._PC4 = pc4
        Batch2048EnvFast._PC16 = pc16
        Batch2048EnvFast._LUT_EMPTY4_ROW = empty4
        Batch2048EnvFast._LUT_MASK_SELECT = lut_sel4
        Batch2048EnvFast._LUT_SELECT16 = lut_sel16

    def _spawn_random_tile_batch_bitwise(self, moved_mask: np.ndarray, p4: float = 0.1):
        """
        보드 단위 16비트 빈칸 플래그(LUT)로 완전 벡터 스폰.
        - moved_mask: (N,) bool
        """
        idx_env = np.nonzero(moved_mask)[0]
        if idx_env.size == 0:
            return

        empty4 = Batch2048EnvFast._LUT_EMPTY4_ROW  # uint8[65536]
        pc16 = Batch2048EnvFast._PC16  # uint8[65536]
        sel16 = Batch2048EnvFast._LUT_SELECT16  # uint8[65536,16,2]

        sub = self._boards[idx_env]  # (M,4) uint16

        # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
        row_masks = empty4[sub]  # (M,4) uint8 (각 행 4bit)
        board_mask16 = (
            (row_masks[:, 0].astype(np.uint16) << 12)
            | (row_masks[:, 1].astype(np.uint16) << 8)
            | (row_masks[:, 2].astype(np.uint16) << 4)
            | (row_masks[:, 3].astype(np.uint16) << 0)
        )  # (M,) uint16

        # 2) 총 빈칸 수 & 유효 보드
        total_empty = pc16[board_mask16.astype(np.int64)].astype(np.int32)  # (M,)
        valid = total_empty > 0
        if not np.any(valid):
            return

        v_rows = sub[valid]  # (Mv,4) uint16
        v_mask16 = board_mask16[valid].astype(np.int64)  # (Mv,) int64 index
        v_tot = total_empty[valid]  # (Mv,)

        # 3) nth & k 샘플
        rng = self._rng
        # numpy Generator.integers는 high에 배열 브로드캐스트 지원. (환경에 따라 random*high로 대체 가능)
        v_nth = rng.integers(0, v_tot, dtype=np.int32)  # (Mv,)
        v_k = np.where(rng.random(size=v_tot.shape) < p4, 2, 1).astype(np.uint16)

        # 4) (row, col) = LUT16[mask16, nth]
        rc = sel16[v_mask16, v_nth]  # (Mv,2) uint8
        rows = rc[:, 0].astype(np.int64)  # (Mv,)
        cols = rc[:, 1].astype(np.uint8)  # (Mv,)

        # 5) 니블 세팅 (scatter)
        shift = (3 - cols).astype(np.uint16) << 2  # (Mv,)
        clear_mask = (~(np.uint16(0xF) << shift) & np.uint16(0xFFFF)).astype(np.uint16)
        write_val = (v_k << shift).astype(np.uint16)

        old_rows = v_rows[np.arange(v_rows.shape[0]), rows]  # (Mv,)
        new_rows = (old_rows & clear_mask) | write_val
        v_rows[np.arange(v_rows.shape[0]), rows] = new_rows

        # 6) 반영
        sub[valid] = v_rows
        self._boards[idx_env] = sub


if __name__ == "__main__":
    import time

    n = 14
    env = Batch2048EnvFast(num_envs=2**n, seed=42)
    obs, info = env.reset()
    tmr = time.time()
    count = 0
    while (time.time() - tmr) < 1:
        _, _, done, _, _ = env.step(0)
        count += 1

    count *= 2**n

    print(f"1초 동안 {count} 에피소드 진행")  # 529680

    # print("Initial boards:")
    # print(obs)
    # print("Info:", info)

    # for step in tqdm.tqdm(range(1000 * 2**20 // env.num_envs)):
    #     actions = env.action_space.sample()
    #     obs, reward, terminated, truncated, info = env.step(actions)

# if __name__ == "__main__":
#     env = Batch2048EnvFast(num_envs=2**13, seed=42)
#     obs, info = env.reset()
#     print("Initial boards:")
#     print(obs)
#     print("Info:", info)

#     for step in tqdm.tqdm(range(1000 * 2**20 // env.num_envs)):
#         actions = env.action_space.sample()
#         obs, reward, terminated, truncated, info = env.step(actions)

# if __name__ == "__main__":
#     env = Batch2048EnvFast(num_envs=1, seed=42)
#     obs, info = env.reset()
#     print("Initial boards:")
#     print(obs)
#     print("Info:", info)

#     for step in range(1000):
#         # actions = env.action_space.sample()
#         input_str = (
#             input("Enter action (a=LEFT, d=RIGHT, w=UP, s=DOWN, q=quit): ")
#             .strip()
#             .lower()
#         )
#         if input_str == "q":
#             break
#         action_map = {"a": 0, "d": 1, "w": 2, "s": 3}
#         if input_str not in action_map:
#             print("Invalid input. Please enter a, d, w, s, or q.")
#             continue
#         actions = np.array([action_map[input_str]], dtype=np.int64)
#         obs, reward, terminated, truncated, info = env.step(actions)
#         print(f"\nStep {step+1}, Actions: {actions}")
#         print("Boards:")
#         for row in obs.swapaxes(0, 1):
#             for r in row:
#                 cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
#                 print(" ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells))
#             print()
