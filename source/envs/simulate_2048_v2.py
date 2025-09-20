# space-only indentation
import numpy as np
import tqdm

class Batch2048EnvSimulator():
    """
    - 내부 상태: boards (N,4) uint16, 각 원소는 4칸 니블(상위→하위)
    - 액션: 0=LEFT, 1=RIGHT, 2=UP, 3=DOWN
    - 스폰/보상 구현. invalid_move, able_move를 info로 제공.
    """

    # 클래스 정적 LUT (좌/우 결과행) — __init__에서 필요 시 1회 생성
    _LUT_LEFT_NEW: np.ndarray | None = None     # uint16[65536]
    _LUT_RIGHT_NEW: np.ndarray | None = None    # uint16[65536]
    _LUT_LEFT_REWARD: np.ndarray | None = None     # int32[65536]
    _LUT_RIGHT_REWARD: np.ndarray | None = None    # int32[65536]

    # 스폰 최적화용 클래스 정적 LUT들
    _PC4: np.ndarray | None = None              # uint8[16], popcount
    _PC16: np.ndarray | None = None             # uint8[65536], popcount
    _LUT_EMPTY4_ROW: np.ndarray | None = None   # uint8[65536], 4비트 빈칸 마스크
    _LUT_MASK_SELECT: np.ndarray | None = None  # uint8[16,4], (mask, nth)->col(0..3) or 255
    _LUT_SELECT16: np.ndarray | None = None     # uint8[65536,16,2], (mask16,nth)->(row,col) or (255,255)
    
    _LUT_TRANSPOSE: np.ndarray | None = None  # uint64[65536]

    _LUT_CHOICE_INDEX: np.ndarray | None = None  # int8[16,4], (mask,nth)->index(0..3) or 0

    _rng: np.random.Generator | None = None  # 난수생성기

    metadata = {"render_modes": []}

    # ---------- 공개 API ----------

    @staticmethod
    def init(seed=2048):
        Batch2048EnvSimulator._rng = np.random.default_rng(seed)

        Batch2048EnvSimulator._LUT_LEFT_NEW, Batch2048EnvSimulator._LUT_RIGHT_NEW, Batch2048EnvSimulator._LUT_LEFT_REWARD, Batch2048EnvSimulator._LUT_RIGHT_REWARD = Batch2048EnvSimulator._build_row_luts()

        Batch2048EnvSimulator._LUT_TRANSPOSE = Batch2048EnvSimulator._build_transpose_luts()

        Batch2048EnvSimulator._init_spawn_luts()

        Batch2048EnvSimulator._LUT_CHOICE_INDEX = Batch2048EnvSimulator._build_choice_luts()

    @staticmethod
    def init_board(n):
        boards = np.zeros((n, 4), dtype=np.uint16)
        Batch2048EnvSimulator.spawn_random_tile(boards, np.full((n,), True), p4=0.1)
        Batch2048EnvSimulator.spawn_random_tile(boards, np.full((n,), True), p4=0.1)
        return boards
    
    @staticmethod
    def move(boards: np.ndarray, action: int | list | tuple | np.ndarray, reward=False):
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
            action = np.full((boards.shape[0],), int(action), dtype=np.int64)

        assert action.shape == (boards.shape[0],)

        rewards = np.zeros((boards.shape[0],), dtype=np.int32)

        # transpose selected boards
        idx_v = np.nonzero((action == 2) | (action == 3))[0]
        if idx_v.size:
            Batch2048EnvSimulator._transpose_inplace(boards, idx_v)

        # UP -> LEFT while transposed, LEFT
        idx = np.nonzero((action == 0) | (action == 2))[0]
        if idx.size:
            if reward:
                Batch2048EnvSimulator._cal_reward(boards, idx, Batch2048EnvSimulator._LUT_LEFT_REWARD, rewards)
            Batch2048EnvSimulator._apply_lut_inplace(boards, idx, Batch2048EnvSimulator._LUT_LEFT_NEW)

        # DOWN -> RIGHT while transposed, RIGHT
        idx = np.nonzero((action == 1) | (action == 3))[0]
        if idx.size:
            if reward:
                Batch2048EnvSimulator._cal_reward(boards, idx, Batch2048EnvSimulator._LUT_RIGHT_REWARD, rewards)
            Batch2048EnvSimulator._apply_lut_inplace(boards, idx, Batch2048EnvSimulator._LUT_RIGHT_NEW)

        # transpose back
        if idx_v.size:
            Batch2048EnvSimulator._transpose_inplace(boards, idx_v)

        return rewards
    
    @staticmethod
    def choice_able_moves(able_move: np.ndarray) -> np.ndarray:
        """
        able_move: (N,4) bool
        각 행마다 True인 열 중 하나를 무작위 선택하여 (N,) int 반환.
        모두 False인 행은 0 반환.
        """
        assert able_move.dtype == bool

        weights = np.array([8, 4, 2, 1])
        lut_choice_index = Batch2048EnvSimulator._LUT_CHOICE_INDEX  # (mask,nth)->index(0..3) or 255

        counts = np.maximum(able_move.sum(axis=1), 1)  # 각 행의 True 개수, 최소 1
        rand_idx = Batch2048EnvSimulator._rng.integers(0, counts)

        move_type = np.dot(able_move, weights)
        output = lut_choice_index[move_type, rand_idx]

        return output

    @staticmethod
    def able_move(boards):
        lut_left = Batch2048EnvSimulator._LUT_LEFT_NEW
        lut_right = Batch2048EnvSimulator._LUT_RIGHT_NEW
        num_envs = boards.shape[0]
        able_move = np.zeros((num_envs, 4), dtype=bool)
        
        boards_left = lut_left[boards]
        boards_right = lut_right[boards]
        able_move[:, 0] = (boards != boards_left).any(axis=1)
        able_move[:, 1] = (boards != boards_right).any(axis=1)

        boards_copy = boards.copy()
        Batch2048EnvSimulator._transpose_inplace(boards_copy)
        boards_up = lut_left[boards_copy]
        boards_down = lut_right[boards_copy]
        able_move[:, 2] = (boards_copy != boards_up).any(axis=1)
        able_move[:, 3] = (boards_copy != boards_down).any(axis=1)

        return able_move

    @staticmethod
    def spawn_random_tile(boards: np.ndarray, moved_mask: np.ndarray, p4: float = 0.1):
        """
        보드 단위 16비트 빈칸 플래그(LUT)로 완전 벡터 스폰.
        - moved_mask: (N,) bool
        """
        idx_env = np.nonzero(moved_mask)[0]
        if idx_env.size == 0:
            return

        empty4 = Batch2048EnvSimulator._LUT_EMPTY4_ROW  # uint8[65536]
        pc16 = Batch2048EnvSimulator._PC16  # uint8[65536]
        sel16 = Batch2048EnvSimulator._LUT_SELECT16  # uint8[65536,16,2]

        sub = boards[idx_env]  # (M,4) uint16

        # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
        row_masks = empty4[sub]  # (M,4) uint8 (각 행 4bit)
        board_mask16 = (
            (row_masks[:, 0].astype(np.uint16) << 12) |
            (row_masks[:, 1].astype(np.uint16) << 8) |
            (row_masks[:, 2].astype(np.uint16) << 4) |
            (row_masks[:, 3].astype(np.uint16) << 0)
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
        rng = Batch2048EnvSimulator._rng
        # numpy Generator.integers는 high에 배열 브로드캐스트 지원. (환경에 따라 random*high로 대체 가능)
        v_nth = rng.integers(0, v_tot, dtype=np.int32)  # (Mv,)
        v_k = np.where(rng.random(size=v_tot.shape) < p4, 2, 1).astype(np.uint16)

        # 4) (row, col) = LUT16[mask16, nth]
        rc = sel16[v_mask16, v_nth]  # (Mv,2) uint8
        rows = rc[:, 0].astype(np.int64)  # (Mv,)
        cols = rc[:, 1].astype(np.uint8)  # (Mv,)

        # 5) 니블 세팅 (scatter)
        shift = ((3 - cols).astype(np.uint16) << 2)  # (Mv,)
        clear_mask = (~(np.uint16(0xF) << shift) & np.uint16(0xFFFF)).astype(np.uint16)
        write_val = (v_k << shift).astype(np.uint16)

        old_rows = v_rows[np.arange(v_rows.shape[0]), rows]  # (Mv,)
        new_rows = (old_rows & clear_mask) | write_val
        v_rows[np.arange(v_rows.shape[0]), rows] = new_rows

        # 6) 반영
        sub[valid] = v_rows
        boards[idx_env] = sub

        return boards

    @staticmethod
    def all_move_boards(boards):
        lut_left = Batch2048EnvSimulator._LUT_LEFT_NEW
        lut_right = Batch2048EnvSimulator._LUT_RIGHT_NEW
        num_envs = boards.shape[0]
        if num_envs == 0:
            return np.array([], dtype=np.uint16), np.array([], dtype=np.int64), np.array([], dtype=np.int64)
        
        moved_boards = np.zeros((num_envs, 4, 4), dtype=np.uint16)
        
        moved_boards[:, 0] = lut_left[boards]
        moved_boards[:, 1] = lut_right[boards]

        boards_copy = boards.copy()
        Batch2048EnvSimulator._transpose_inplace(boards_copy)
        up_boards = lut_left[boards_copy]
        down_boards = lut_right[boards_copy]
        Batch2048EnvSimulator._transpose_inplace(up_boards)
        Batch2048EnvSimulator._transpose_inplace(down_boards)
        moved_boards[:, 2] = up_boards
        moved_boards[:, 3] = down_boards

        mask = (moved_boards != np.expand_dims(boards, axis=1)).any(axis=2)

        return moved_boards[mask], np.nonzero(mask)[0], np.nonzero(mask)[1]


    @staticmethod
    def all_next_boards(boards, only_2=False):
        if boards.shape[0] == 0:
            return np.array([], dtype=np.uint16), np.array([], dtype=np.int64), np.array([], dtype=np.uint8)
        
        empty4 = Batch2048EnvSimulator._LUT_EMPTY4_ROW  # uint8[65536]
        pc16 = Batch2048EnvSimulator._PC16  # uint8[65536]
        sel16 = Batch2048EnvSimulator._LUT_SELECT16  # uint8[65536,16,2]

        # 1) 행 마스크 → 보드 마스크 16비트 (벡터)
        row_masks = empty4[boards]  # (M,4) uint8 (각 행 4bit)
        board_mask16 = (
            (row_masks[:, 0].astype(np.uint16) << 12) |
            (row_masks[:, 1].astype(np.uint16) << 8) |
            (row_masks[:, 2].astype(np.uint16) << 4) |
            (row_masks[:, 3].astype(np.uint16) << 0)
        )  # (M,) uint16

        # 2) 총 빈칸 수 & 유효 보드
        total_empty = pc16[board_mask16.astype(np.int64)].astype(np.int32)  # (M,)
        valid = total_empty > 0
        if not np.any(valid):
            return np.array([], dtype=np.uint16), np.array([], dtype=np.int64), np.array([], dtype=np.uint8)

        v_rows = boards[valid]  # (Mv,4) uint16
        v_mask16 = board_mask16[valid].astype(np.int64)  # (Mv,) int64 index
        v_tot = total_empty[valid]  # (Mv,)

        mask = np.arange(16) < v_tot[:, None]
        rc = sel16[v_mask16][mask]  # (Mu,2) uint8
        rows = rc[:, 0].astype(np.int64)  # (Mu,)
        cols = rc[:, 1].astype(np.uint8)  # (Mu,)

        idx = np.nonzero(mask)[0]  # (Mu,)
        if not only_2:
            idx = idx.repeat(2)
        out = v_rows[idx].copy() 

        # 5) 니블 세팅 (scatter)
        shift = ((3 - cols).astype(np.uint16) << 2)
        clear_mask = (~(np.uint16(0xF) << shift) & np.uint16(0xFFFF)).astype(np.uint16)
        write_val_2 = (1 << shift).astype(np.uint16)
        if not only_2:
            write_val_4 = (2 << shift).astype(np.uint16)
        
        if only_2:
            old_rows = out[np.arange(out.shape[0]), rows]
            new_rows = (old_rows & clear_mask) | write_val_2
            out[np.arange(out.shape[0]), rows] = new_rows
            return out, idx, v_tot
        
        else:
            row_idx = np.arange(0, idx.shape[0], 2)
            old_rows = out[row_idx, rows]
            new_rows = (old_rows & clear_mask) | write_val_2
            out[row_idx, rows] = new_rows

            row_idx += 1
            old_rows = out[row_idx, rows]
            new_rows = (old_rows & clear_mask) | write_val_4
            out[row_idx, rows] = new_rows

            return out, idx, v_tot * 2

    # ---------- 유틸 (모두 클래스/인스턴스 메소드, 전역 없음) ----------

    @staticmethod
    def _pack_row(vals: np.ndarray) -> int:
        # vals: (4,) uint8  [a b c d] (a가 상위니블)
        return (int(vals[0]) << 12) | (int(vals[1]) << 8) | (int(vals[2]) << 4) | int(vals[3])

    @staticmethod
    def _unpack_row(r: int) -> np.ndarray:
        return np.array([(r >> 12) & 0xF, (r >> 8) & 0xF, (r >> 4) & 0xF, r & 0xF], dtype=np.uint8)

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
                reward += (1 << (comp[i] + 1))
                i += 2
            else:
                out.append(comp[i]); i += 1
        while len(out) < 4:
            out.append(0)
        return np.uint16(cls._pack_row(np.minimum(np.array(out[:4], dtype=np.uint8), 15))), reward

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
            return ((r & 0x000F) << 12) | ((r & 0x00F0) << 4) | ((r & 0x0F00) >> 4) | ((r & 0xF000) >> 12)

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
            lut_trans[r] = (r & 0xF000) | (r & 0x0F00) << 20 | (r & 0x00F0) << 40 | (r & 0x000F) << 60

        return lut_trans
    
    @classmethod
    def _build_choice_luts(cls):
        lut_choice_index = np.zeros((16, 4), dtype=np.uint8)
        for r in range(16):
            count = 0
            for i in range(4):
                if r & (1 << (3 - i)):
                    lut_choice_index[r, count] = i
                    count += 1

        return lut_choice_index

    @staticmethod
    def _apply_lut_inplace(boards, idx: np.ndarray, lut_rows: np.ndarray):
        """
        선택된 보드 idx의 4개 행에 대해 주어진 수평 LUT를 적용.
        행마다 독립적으로 1D 인덱싱 → 벡터화.
        """
        boards[idx] = lut_rows[boards[idx]]

    @staticmethod
    def _cal_reward(boards, idx: np.ndarray, lut_reward: np.ndarray, reward: np.ndarray):
        """
        선택된 보드 idx의 4개 행에 대해 주어진 수평 보상 LUT를 적용하여 보상 합산.
        행마다 독립적으로 1D 인덱싱 → 벡터화.
        """
        reward[idx] = lut_reward[boards[idx]].sum(axis=1)

    @staticmethod
    def _transpose_inplace(boards, idx: np.ndarray | None = None):
        """
        비트연산 전치 (4x4 니블).
        boards[idx]: (M,4) uint16 의 각 보드를 전치하여 다시 (M,4)에 저장.
        """
        if idx is None:
            sub = boards
        else:
            sub = boards[idx]  # (M,4), 각 행은 0xABCD 니블들

        lut_trans = Batch2048EnvSimulator._LUT_TRANSPOSE
        t = lut_trans[sub[:, 0]] | (lut_trans[sub[:, 1]] >> 4) | (lut_trans[sub[:, 2]] >> 8) | (lut_trans[sub[:, 3]] >> 12)

        if idx is None:
            boards[:, 0] = t
            boards[:, 1] = (t >> 16)
            boards[:, 2] = (t >> 32)
            boards[:, 3] = (t >> 48)
        else:
            # use implicit type casting
            boards[idx, 0] = t
            boards[idx, 1] = (t >> 16)
            boards[idx, 2] = (t >> 32)
            boards[idx, 3] = (t >> 48)

        # same as this code
        # row_mask = np.uint64(0xFFFF)
        # self._boards[idx, 0] = t & row_mask
        # self._boards[idx, 1] = (t >> 16) & row_mask
        # self._boards[idx, 2] = (t >> 32) & row_mask
        # self._boards[idx, 3] = (t >> 48) & row_mask

    def a():
        pass


    @classmethod
    def _init_spawn_luts(cls):
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
        lut_sel16 = np.full((1 << 16, 16, 2), 255, dtype=np.uint8)  # (row,col) = (255,255) = invalid
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

        Batch2048EnvSimulator._PC4 = pc4
        Batch2048EnvSimulator._PC16 = pc16
        Batch2048EnvSimulator._LUT_EMPTY4_ROW = empty4
        Batch2048EnvSimulator._LUT_MASK_SELECT = lut_sel4
        Batch2048EnvSimulator._LUT_SELECT16 = lut_sel16

Batch2048EnvSimulator.init()
    

if __name__ == "__main__":
    num_env = 2
    boards = Batch2048EnvSimulator.init_board(num_env)
    able_move = Batch2048EnvSimulator.able_move(boards)

    # for step in tqdm.tqdm(range(100*2**20//num_env)):
    #     actions = Batch2048EnvSimulator.choice_able_moves(able_move)
    #     rewards = Batch2048EnvSimulator.move(boards, actions, reward=True)
    #     mask = actions == 255
    #     Batch2048EnvSimulator.spawn_random_tile(boards, mask)

    moved_boards, index, move = Batch2048EnvSimulator.all_move_boards(boards)
    print(moved_boards, index, move)
