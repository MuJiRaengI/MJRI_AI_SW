# space-only indentation
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import tqdm
import ray
from ray.experimental import tqdm_ray


class Batch2048EnvFast(gym.Env):
    """
    - 내부 상태: boards (N,4) uint16, 각 원소는 4칸 니블(상위→하위)
    - 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    - 스폰/보상 구현. invalid_move, able_move를 info로 제공.
    """

    # 클래스 정적 LUT (좌/우 결과행) — __init__에서 필요 시 1회 생성
    _LUT_LEFT_NEW: np.ndarray | None = None  # uint16[65536]
    _LUT_RIGHT_NEW: np.ndarray | None = None  # uint16[65536]
    _LUT_LEFT_REWARD: np.ndarray | None = None  # int32[65536]
    _LUT_RIGHT_REWARD: np.ndarray | None = None  # int32[65536]

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

    _LUT_TRANSPOSE: np.ndarray | None = None  # uint64[65536]

    _LUT_CHOICE_INDEX: np.ndarray | None = (
        None  # int8[16,4], (mask,nth)->index(0..3) or 255
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
            (
                Batch2048EnvFast._LUT_LEFT_NEW,
                Batch2048EnvFast._LUT_RIGHT_NEW,
                Batch2048EnvFast._LUT_LEFT_REWARD,
                Batch2048EnvFast._LUT_RIGHT_REWARD,
            ) = self._build_row_luts()

        if Batch2048EnvFast._LUT_TRANSPOSE is None:
            Batch2048EnvFast._LUT_TRANSPOSE = self._build_transpose_luts()

        # 스폰용 LUT들 준비 (최초 한 번)
        if Batch2048EnvFast._LUT_EMPTY4_ROW is None:
            self._init_spawn_luts()

        if Batch2048EnvFast._LUT_CHOICE_INDEX is None:
            Batch2048EnvFast._LUT_CHOICE_INDEX = self._build_choice_luts()

    # ---------- 공개 API ----------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._boards.fill(0)
        # Removed self._is_transposed.fill(False)
        self._spawn_random_tile_batch_bitwise(np.full((self.num_envs,), True), p4=0.1)
        self._spawn_random_tile_batch_bitwise(np.full((self.num_envs,), True), p4=0.1)
        obs = self._boards.copy()
        info = {
            "invalid_move": np.zeros(
                (self.num_envs,), dtype=bool
            ),  # 이번 액션이 무효였는지
            "able_move": self._able_move(),  # 각 방향으로 이동 가능 여부 (N,4) bool
        }
        return obs, info

    def step(self, action: int | list | tuple | np.ndarray):
        """
        action: (N,) int64 in {0,1,2,3}
        - 수평 액션(0/1)인데 현재 전치 상태면 → 전치 해제(원상태로)
        - 수직 액션(2/3)인데 현재 비전치면 → 전치 적용(수평화)
        그 뒤, 좌/우 LUT로 행별 변환.
        """
        if isinstance(action, (list, tuple)):
            action = np.asarray(action, dtype=np.int64)
        elif isinstance(action, np.ndarray):
            action = action.astype(np.int64)
        else:
            action = np.full((self.num_envs,), int(action), dtype=np.int64)

        assert action.shape == (self.num_envs,)

        # Removed transpose state alignment block using _is_transposed

        before = self._boards.copy()
        reward = np.zeros((self.num_envs,), dtype=np.int32)

        # transpose selected boards
        idx_v = np.nonzero((action == 2) | (action == 3))[0]
        if idx_v.size:
            self._transpose_inplace(idx_v)

        # UP -> LEFT while transposed, LEFT
        idx = np.nonzero((action == 0) | (action == 2))[0]
        if idx.size:
            self._cal_reward(idx, Batch2048EnvFast._LUT_LEFT_REWARD, reward)
            self._apply_lut_inplace(idx, Batch2048EnvFast._LUT_LEFT_NEW)

        # DOWN -> RIGHT while transposed, RIGHT
        idx = np.nonzero((action == 1) | (action == 3))[0]
        if idx.size:
            self._cal_reward(idx, Batch2048EnvFast._LUT_RIGHT_REWARD, reward)
            self._apply_lut_inplace(idx, Batch2048EnvFast._LUT_RIGHT_NEW)

        # transpose back
        if idx_v.size:
            self._transpose_inplace(idx_v)

        # 3) 이동된 보드에 대해 타일 스폰
        invalid = (self._boards == before).all(axis=1)
        moved_mask = ~invalid
        self._spawn_random_tile_batch_bitwise(moved_mask, p4=0.1)

        # 4) 종료, 보상, 관측값, info 반환
        obs = self._boards.copy()
        able_move = self._able_move()
        # done if no able move in all direction
        # terminated = ~(able_move.any(axis=1))
        terminated = ~(able_move.any(axis=1)) | invalid

        # terminated = np.zeros((self.num_envs,), dtype=bool)
        # no time limit
        truncated = np.zeros((self.num_envs,), dtype=bool)
        info = {
            "invalid_move": invalid,  # 이번 액션이 무효였는지
            "able_move": able_move,  # 각 방향으로 이동 가능 여부 (N,4) bool
        }
        return obs, reward, terminated, truncated, info

    def choice_able_moves(self, able_move: np.ndarray) -> np.ndarray:
        """
        able_move: (N,4) bool
        각 행마다 True인 열 중 하나를 무작위 선택하여 (N,) int 반환.
        모두 False인 행은 255 반환.
        """
        assert able_move.shape == (self.num_envs, 4) and able_move.dtype == bool

        weights = np.array([8, 4, 2, 1])
        lut_choice_index = (
            Batch2048EnvFast._LUT_CHOICE_INDEX
        )  # (mask,nth)->index(0..3) or 255

        counts = np.maximum(able_move.sum(axis=1), 1)  # 각 행의 True 개수, 최소 1
        rand_idx = self._rng.integers(0, counts)

        move_type = np.dot(able_move, weights)
        output = lut_choice_index[move_type, rand_idx]

        return output

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
        # 보상, 결과 행만 반환 (LUT 용)
        comp = [int(v) for v in vals if v != 0]
        out = []
        i = 0
        reward = 0
        while i < len(comp):
            if i + 1 < len(comp) and comp[i] == comp[i + 1]:
                out.append(comp[i] + 1)
                reward += 1 << (comp[i] + 1)
                i += 2
            else:
                out.append(comp[i])
                i += 1
        while len(out) < 4:
            out.append(0)
        return (
            np.uint16(cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15))),
            reward,
        )

    @classmethod
    def _build_row_luts(cls):
        """
        좌/우 결과, 보상 행 LUT 생성.
        - LEFT:  r -> left(r)
        - RIGHT: r -> reverse(left(reverse(r)))  (빌드 타임에 계산해 런타임 reverse 제거)
        """
        lut_left = np.zeros(65536, dtype=np.uint16)
        lut_right = np.zeros(65536, dtype=np.uint16)
        lut_left_reward = np.zeros(65536, dtype=np.int32)
        lut_right_reward = np.zeros(65536, dtype=np.int32)

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
            left_r, reward = cls._slide_merge_left_row(orig)
            lut_left[r] = left_r
            lut_left_reward[r] = reward

            # RIGHT (빌드 타임에 역방향 LUT 고정)
            rev = reverse_row16(r)
            rev_orig = cls._unpack_row(rev)
            rev_left, reward = cls._slide_merge_left_row(rev_orig)
            right_r = reverse_row16(int(rev_left))
            lut_right[r] = right_r
            lut_right_reward[r] = reward

        return lut_left, lut_right, lut_left_reward, lut_right_reward

    @classmethod
    def _build_transpose_luts(cls):
        lut_trans = np.zeros(65536, dtype=np.uint64)
        for r in range(65536):
            lut_trans[r] = (
                (r & 0xF000)
                | (r & 0x0F00) << 20
                | (r & 0x00F0) << 40
                | (r & 0x000F) << 60
            )

        return lut_trans

    @classmethod
    def _build_choice_luts(cls):
        lut_choice_index = np.full((16, 4), 255, dtype=np.uint8)
        for r in range(16):
            count = 0
            for i in range(4):
                if r & (1 << (3 - i)):
                    lut_choice_index[r, count] = i
                    count += 1

        return lut_choice_index

    def _apply_lut_inplace(self, idx: np.ndarray, lut_rows: np.ndarray):
        """
        선택된 보드 idx의 4개 행에 대해 주어진 수평 LUT를 적용.
        행마다 독립적으로 1D 인덱싱 → 벡터화.
        """
        self._boards[idx] = lut_rows[self._boards[idx]]

    def _cal_reward(self, idx: np.ndarray, lut_reward: np.ndarray, reward: np.ndarray):
        """
        선택된 보드 idx의 4개 행에 대해 주어진 수평 보상 LUT를 적용하여 보상 합산.
        행마다 독립적으로 1D 인덱싱 → 벡터화.
        """
        reward[idx] = lut_reward[self._boards[idx]].sum(axis=1)

    def _able_move(self):
        lut_left = Batch2048EnvFast._LUT_LEFT_NEW
        lut_right = Batch2048EnvFast._LUT_RIGHT_NEW
        able_move = np.zeros((self.num_envs, 4), dtype=bool)

        boards = self._boards.copy()
        boards_left = lut_left[self._boards]
        boards_right = lut_right[self._boards]
        able_move[:, 0] = (self._boards != boards_left).any(axis=1)
        able_move[:, 1] = (self._boards != boards_right).any(axis=1)

        self._transpose_inplace()
        boards_up = lut_left[self._boards]
        boards_down = lut_right[self._boards]
        able_move[:, 2] = (self._boards != boards_up).any(axis=1)
        able_move[:, 3] = (self._boards != boards_down).any(axis=1)

        self._boards = boards
        return able_move

    def _transpose_inplace(self, idx: np.ndarray | None = None):
        """
        비트연산 전치 (4x4 니블).
        boards[idx]: (M,4) uint16 의 각 보드를 전치하여 다시 (M,4)에 저장.
        """
        if idx is None:
            sub = self._boards
        else:
            sub = self._boards[idx]  # (M,4), 각 행은 0xABCD 니블들

        lut_trans = self._LUT_TRANSPOSE
        t = (
            lut_trans[sub[:, 0]]
            | (lut_trans[sub[:, 1]] >> 4)
            | (lut_trans[sub[:, 2]] >> 8)
            | (lut_trans[sub[:, 3]] >> 12)
        )

        if idx is None:
            self._boards[:, 0] = t
            self._boards[:, 1] = t >> 16
            self._boards[:, 2] = t >> 32
            self._boards[:, 3] = t >> 48
        else:
            # use implicit type casting
            self._boards[idx, 0] = t
            self._boards[idx, 1] = t >> 16
            self._boards[idx, 2] = t >> 32
            self._boards[idx, 3] = t >> 48

        # same as this code
        # row_mask = np.uint64(0xFFFF)
        # self._boards[idx, 0] = t & row_mask
        # self._boards[idx, 1] = (t >> 16) & row_mask
        # self._boards[idx, 2] = (t >> 32) & row_mask
        # self._boards[idx, 3] = (t >> 48) & row_mask

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


@ray.remote
def test(steps, bar):
    env = Batch2048EnvFast(num_envs=2**15, seed=42)
    obs, info = env.reset()

    for step in range(steps):
        actions = env.choice_able_moves(info["able_move"])
        obs, reward, terminated, truncated, info = env.step(actions)
        if (step + 1) % 100 == 0:
            bar.update.remote(100)


# if __name__ == "__main__":
#     env = Batch2048EnvFast(num_envs=2**15, seed=42)
#     obs, info = env.reset()
#     print("Initial boards:")
#     print(obs)
#     print("Info:", info)
#     remote_tqdm = ray.remote(tqdm_ray.tqdm)

#     for step in tqdm.tqdm(range(200*2**20//env.num_envs)):
#         # actions = env.action_space.sample()
#         # obs, reward, terminated, truncated, info = env.step(actions)
#         actions = env.choice_able_moves(info["able_move"])
#         obs, reward, terminated, truncated, info = env.step(actions)
#         if terminated.all():
#             env.reset()

#     # # with multi processing
#     # n = 1000*2**20//env.num_envs
#     # split = 20  # cpu core num

#     # ray.init(num_cpus=split)
#     # remote_tqdm = ray.remote(tqdm_ray.tqdm)
#     # bar = remote_tqdm.remote(total=n)
#     # obj = []
#     # for i in range(split):
#     #     step = n // split if i != 0 else n // split + n % split
#     #     obj.append(test.remote(step, bar))

#     # ray.get(obj)
#     # bar.close.remote()


if __name__ == "__main__":
    env = Batch2048EnvFast(num_envs=1, seed=42)
    obs, info = env.reset()
    print("Initial boards:")
    for row in obs.swapaxes(0, 1):
        for r in row:
            cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
            print(" ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells))
        print()
    print("Info:", info)

    for step in range(1000):
        # actions = env.action_space.sample()
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
        actions = np.array([action_map[input_str]], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(actions)
        print(f"\nStep {step+1}, Actions: {actions}, Reward: {reward}")
        print("Info:", info)
        print("Boards:")
        for row in obs.swapaxes(0, 1):
            for r in row:
                cells = [(r >> shift) & 0xF for shift in (12, 8, 4, 0)]
                print(" ".join(f"{(1 << v) if v > 0 else 0:4d}" for v in cells))
            print()
