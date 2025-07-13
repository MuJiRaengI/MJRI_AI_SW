import gym
from gym import spaces
import numpy as np
import pygame
import torch
from PIL import Image
from collections import deque
import cv2
import torch.nn.functional as F
import os


class EnvAvoidObserver(gym.Env):
    def __init__(
        self,
        map_size_px=(4096, 4096),
        tile_size=64,
        max_steps=1000,
        num_observers=0,
        h=640,
        w=1280,
        scale_factor=0.25,
        frame_stack=4,
        random_start=False,
        move_observer=False,
    ):
        super().__init__()
        self.map_size_px = np.array(map_size_px)
        self.tile_size = tile_size
        self.max_steps = max_steps
        self.num_observers = num_observers
        self.scale_factor = scale_factor
        self.frame_stack = frame_stack
        self.random_start = random_start
        self.move_observer = move_observer
        self._rng = np.random.RandomState(None)  # 인스턴스별 난수 생성기
        self.h = int(h * self.scale_factor)
        self.w = int(w * self.scale_factor)

        self.x_crop, self.y_crop, self.w_crop, self.h_crop = 400, 100, 512, 512

        self.camera_size_tiles = (20, 10)
        self.camera_size_px = (
            self.camera_size_tiles[0] * tile_size,
            self.camera_size_tiles[1] * tile_size,
        )

        # Speeds
        self.agent_speed = 20  # agent is twice observer speed
        self.observer_speed = 10.0
        self.observer_pause_frames = 30  # 1초 대기 (30fps 기준)

        obs_dim = 2 + 2 * num_observers
        # observation_space를 Dict로 정의
        self.observation_space = spaces.Dict(
            {
                # "image": spaces.Box(
                #     low=0,
                #     high=255,
                #     shape=(3 * self.frame_stack, self.h, self.w),
                #     dtype=np.uint8,
                # ),
                "image": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        3 * self.frame_stack,
                        int(self.h_crop * scale_factor),
                        int(self.w_crop * scale_factor),
                    ),
                    dtype=np.uint8,
                ),
                "vector": spaces.Box(
                    low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
                ),
            }
        )
        self.action_space = spaces.Discrete(8)

        self.state_buffer = deque(maxlen=self.frame_stack)

        self.record = False
        self.video_writer = None

        self._init_obstacle_mask()
        self._build_background_surface()
        self._load_direction_map()
        self.reset()

    def _init_obstacle_mask(self):
        image_path = os.path.join(os.path.dirname(__file__), "map.png")
        img = Image.open(image_path).convert("RGB").resize(self.map_size_px)
        img_array = np.array(img)
        blue_mask = (
            (img_array[:, :, 2] > 180)
            & (img_array[:, :, 0] < 100)
            & (img_array[:, :, 1] < 150)
        )
        self.obstacle_mask = blue_mask.astype(np.uint8)

        safe_rgb = np.array([40, 155, 110])
        safe_mask = np.all(img_array == safe_rgb, axis=-1)
        self.safe_mask = safe_mask.astype(np.uint8)

        goal_rgb = np.array([157, 92, 187])
        goal_mask = np.all(img_array == goal_rgb, axis=-1)
        self.goal_mask = goal_mask.astype(np.uint8)

    def _load_direction_map(self):
        path = os.path.join(os.path.dirname(__file__), "map_direct.png")
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise FileNotFoundError(f"Failed to load direct map from {path}")
        if gray.shape != tuple(self.map_size_px[::-1]):
            raise ValueError(
                f"Direct map shape mismatch: expected {self.map_size_px[::-1]}, got {gray.shape}"
            )
        self.direct_map = gray

    def _build_background_surface(self):
        h, w = self.obstacle_mask.shape
        bg_arr = np.zeros((h, w, 3), dtype=np.uint8)
        bg_arr[self.obstacle_mask == 1] = [0, 0, 128]
        bg_arr[self.safe_mask == 1] = [40, 155, 110]
        bg_arr[self.goal_mask == 1] = [157, 92, 187]
        self.background = pygame.surfarray.make_surface(bg_arr.transpose((1, 0, 2)))

    def _is_safe(self, pos) -> bool:
        pos = np.asarray(pos, dtype=int)
        x, y = pos
        h, w = self.safe_mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.safe_mask[y, x] == 1
        return False

    def _is_goal(self, pos) -> bool:
        pos = np.asarray(pos, dtype=int)
        x, y = pos
        h, w = self.goal_mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.goal_mask[y, x] == 1
        return False

    def reset(self, *args, **kwargs):
        if kwargs.get("seed") is not None:
            self._rng = np.random.RandomState(kwargs["seed"])

        self.steps = 0
        self.total_reward = 0.0
        self.reward = 0.0
        self.last_action = None

        if self.random_start:
            for _ in range(100):
                start_tile = self._rng.randint(10, 55, size=2)
                self.agent_pos = (start_tile * self.tile_size).astype(np.float32)
                if not self._is_obstacle(self.agent_pos):
                    break
        else:
            start_tile = np.array([12, 53])
            self.agent_pos = (start_tile * self.tile_size).astype(np.float32)

        self.observers = []
        self.observer_targets = []
        self.observer_waits = []
        for _ in range(self.num_observers):
            while True:
                pos = self._rng.randint(0, self.map_size_px[0], size=2).astype(
                    np.float32
                )
                # 안전지역에 위치하면 다시 위치를 랜덤으로 잡음
                if not self._is_safe(pos):
                    break
            target = self._rng.randint(0, self.map_size_px[0], size=2).astype(
                np.float32
            )
            self.observers.append(pos)
            self.observer_targets.append(target)
            self.observer_waits.append(0)

        state = self.get_state_tensor()
        self.state_buffer.extend([state] * 3)

        return self._get_obs()

    def step(self, action: int):
        self.last_action = action
        if not hasattr(self, "total_reward"):
            self.total_reward = 0.0

        direction = np.array([0, 0])
        if action is not None:
            direction_map = {
                0: np.array([1, 0]),
                1: np.array([1, 1]),
                2: np.array([0, 1]),
                3: np.array([-1, 1]),
                4: np.array([-1, 0]),
                5: np.array([-1, -1]),
                6: np.array([0, -1]),
                7: np.array([1, -1]),
            }
            direction = direction_map.get(action, np.array([0, 0]))
            if np.any(direction != 0):
                direction = direction / np.linalg.norm(direction)

        # 이동 시도
        new_pos = self.agent_pos + direction * self.agent_speed
        new_pos = np.clip(new_pos, [0, 0], self.map_size_px - 1)

        # 벽(강) 충돌 검사: 이동하기 전에 확인
        if self._is_obstacle(new_pos):
            self.reward = self.calc_reward(action, pos=new_pos)
            self.total_reward += self.reward
            self.steps += 1
            image_obs = self._get_image_obs()  # (9, h, w) numpy array 반환
            vector_obs = self._get_vector_obs()  # (4,) numpy array 반환
            obs = {
                "image": np.array(image_obs, dtype=np.uint8),
                "vector": np.array(vector_obs, dtype=np.float32),
            }
            return obs, self.reward, True, {}

        # 위치 갱신
        self.agent_pos = new_pos

        # 옵저버 이동
        if self.move_observer:
            for i in range(self.num_observers):
                if self.observer_waits[i] > 0:
                    self.observer_waits[i] -= 1
                    continue
                obs = self.observers[i]
                tgt = self.observer_targets[i]
                vec = tgt - obs
                dist = np.linalg.norm(vec)
                if dist < self.observer_speed:
                    self.observers[i] = tgt
                    self.observer_waits[i] = self.observer_pause_frames
                    self.observer_targets[i] = self._rng.randint(
                        0, self.map_size_px[0], size=2
                    ).astype(np.float32)
                else:
                    direction = vec / dist
                    cand = obs + direction * self.observer_speed
                    self.observers[i] = np.clip(cand, [0, 0], self.map_size_px - 1)

        # 종료 조건 판별
        done = (
            self._is_goal(self.agent_pos)
            or self._check_collision()
            or self.steps >= self.max_steps
        )

        self.reward = self.calc_reward(action)
        self.total_reward += self.reward
        self.steps += 1

        state = self.get_state_tensor()
        self.state_buffer.append(state)

        return self._get_obs(), self.reward, done, {}

    def calc_reward(self, action, pos=None):
        if pos is None:
            pos = self.agent_pos
        if self._is_goal(pos):
            return 1.0
        elif self._is_obstacle(pos) or self._check_collision():
            return -10.0

        # direction = np.array(self.get_direction_one_hot()).argmax()
        # degree = [np.pi / 4 * i for i in range(8)]
        # reward = np.cos(degree[direction * 2] - degree[action]) * 0.2 - 0.05
        reward = 0.05

        # reward = 0.0

        # direction = np.array(self.get_direction_one_hot()).argmax()
        # # 0 : 오른쪽
        # # 1 : 오른쪽 아래
        # # 2 : 아래
        # # 3 : 왼쪽 아래
        # # 4 : 왼쪽
        # # 5 : 왼쪽 위
        # # 6 : 위
        # # 7 : 오른쪽 위
        # if direction == 0:  # right
        #     if action in [0, 1, 7]:
        #         reward = 0.2
        #     # elif action in [2, 6]:
        #     #     reward = 0.0
        # elif direction == 1:  # down
        #     if action in [1, 2, 3]:
        #         reward = 0.2
        #     # elif action in [0, 4]:
        #     #     reward = 0.0
        # elif direction == 2:  # left
        #     if action in [3, 4, 5]:
        #         reward = 0.2
        #     # elif action in [2, 6]:
        #     #     reward = 0.0
        # elif direction == 3:  # up
        #     if action in [5, 6, 7]:
        #         reward = 0.2
        #     # elif action in [0, 4]:
        #     #     reward = 0.0

        # # ======= 거리 기반 보상 계산 ========
        # # 현재 에이전트 좌표
        # x, y = pos.astype(int)
        # # 크롭할 영역 계산
        # crop_size = 256
        # half_size = crop_size // 2
        # x_start, y_start = max(0, x - half_size), max(0, y - half_size)
        # x_end, y_end = min(self.map_size_px[0], x + half_size), min(
        #     self.map_size_px[1], y + half_size
        # )

        # # 크롭된 마스크
        # cropped_mask = self.obstacle_mask[y_start:y_end, x_start:x_end]

        # # 옵저버 탐색 (옵저버도 동일한 크롭)
        # cropped_observers = [
        #     observer
        #     for observer in self.observers
        #     if x_start <= observer[0] < x_end and y_start <= observer[1] < y_end
        # ]

        # # 에이전트 기준으로 상대 좌표 변환
        # agent_local_pos = np.array([x - x_start, y - y_start])

        # # 옵저버와의 최소 거리
        # if cropped_observers:
        #     min_dist_observer = min(
        #         np.linalg.norm(agent_local_pos - (observer - [x_start, y_start]))
        #         for observer in cropped_observers
        #     )
        # else:
        #     min_dist_observer = float("inf")

        # # 강가와의 최소 거리
        # river_indices = np.argwhere(cropped_mask == 1)
        # if len(river_indices) > 0:
        #     min_dist_river = np.min(
        #         np.linalg.norm(river_indices - agent_local_pos, axis=1)
        #     )
        # else:
        #     min_dist_river = float("inf")

        # # 최종 거리 계산
        # min_dist = min(min_dist_observer, min_dist_river)

        # # ======== 거리 기반 보상 계산 ========
        # linear_penalty = 0.0
        # if min_dist <= 50:
        #     linear_penalty = -0.5
        # elif 50 < min_dist < 100:
        #     # 거리 비율을 0 ~ 1로 정규화
        #     distance_ratio = (min_dist - 50) / 50
        #     # 지수 함수 형태로 패널티 감소 (0.5 * exp(-3 * x))
        #     linear_penalty = -0.5 * np.exp(-3 * distance_ratio) - 0.01

        # reward += linear_penalty
        # # ==================================

        return reward

    def get_stacked_state(self):
        return F.interpolate(
            torch.concat(list(self.state_buffer), dim=0).unsqueeze(0),
            scale_factor=(self.scale_factor, self.scale_factor),
            mode="nearest",
        ).squeeze()

    def _is_obstacle(self, pos) -> bool:
        pos = np.asarray(pos, dtype=int)
        x, y = pos
        h, w = self.obstacle_mask.shape
        if 0 <= x < w and 0 <= y < h:
            return self.obstacle_mask[y, x] == 1
        return True

    def _check_collision(self) -> bool:
        if self._is_safe(self.agent_pos):
            return False
        return any(
            np.linalg.norm(self.agent_pos - obs) < 32.0 for obs in self.observers
        )

    def _get_obs(self):
        # obs 생성
        image_obs = self._get_image_obs()
        vector_obs = self._get_vector_obs()
        obs = {
            "image": np.array(image_obs, dtype=np.uint8),
            "vector": np.array(vector_obs, dtype=np.float32),
        }
        return obs

    def get_state_tensor(self):
        sw, sh = self.camera_size_px
        cam_x = int(self.agent_pos[0] - sw // 2)
        cam_y = int(self.agent_pos[1] - sh // 2)
        cam_x = np.clip(cam_x, 0, self.map_size_px[0] - sw)
        cam_y = np.clip(cam_y, 0, self.map_size_px[1] - sh)
        crop = np.s_[cam_y : cam_y + sh, cam_x : cam_x + sw]

        obs_mask = self.obstacle_mask[crop].copy()
        obs_mask = obs_mask[np.newaxis, :, :]

        observer_mask = np.zeros((sh, sw), dtype=np.uint8)
        agent_mask = np.zeros((sh, sw), dtype=np.uint8)

        def draw_circle(mask, cx, cy, radius=16):
            y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            circle = x**2 + y**2 <= radius**2
            ys, xs = circle.nonzero()
            for dy, dx in zip(ys - radius, xs - radius):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < sh and 0 <= nx < sw:
                    mask[ny, nx] = 1

        for obs in self.observers:
            screen_pos = obs - np.array([cam_x, cam_y])
            ox, oy = screen_pos.astype(int)
            if 0 <= ox < sw and 0 <= oy < sh:
                draw_circle(observer_mask, ox, oy, radius=16)

        ax, ay = (self.agent_pos - [cam_x, cam_y]).astype(int)
        draw_circle(agent_mask, ax, ay, radius=16)

        observer_mask = observer_mask[np.newaxis, :, :]
        agent_mask = agent_mask[np.newaxis, :, :]

        scene = torch.from_numpy(
            np.concatenate([obs_mask, observer_mask, agent_mask], axis=0)
        )

        scene = scene[
            :,
            self.y_crop : self.y_crop + self.h_crop,
            self.x_crop : self.x_crop + self.w_crop,
        ]

        return scene * 255

    def get_direction_one_hot(self, pos=None):
        """
        Args:
            pos: (x, y) 실수 또는 정수 위치
        Returns:
            One-hot direction vector: [right, down, left, up]
        """
        if pos is None:
            pos = self.agent_pos
        x, y = np.asarray(pos, dtype=int)
        h, w = self.direct_map.shape
        if not (0 <= x < w and 0 <= y < h):
            return [0, 0, 0, 0]

        val = self.direct_map[y, x]
        if val == 0:
            return [1, 0, 0, 0]  # right
        elif val == 50:
            return [0, 1, 0, 0]  # down
        elif val == 100:
            return [0, 0, 1, 0]  # left
        elif val == 150:
            return [0, 0, 0, 1]  # up
        else:
            return [0, 0, 0, 0]  # unknown or river

    def render(self, mode="human", logits=None):
        try:
            sw, sh = self.camera_size_px
            if not hasattr(self, "screen"):
                pygame.init()
                self.screen = pygame.display.set_mode((sw, sh))
                pygame.display.set_caption("Pixel-Based Avoid Observer")
                self.clock = pygame.time.Clock()

            cam_x = int(self.agent_pos[0] - sw // 2)
            cam_y = int(self.agent_pos[1] - sh // 2)
            cam_x = np.clip(cam_x, 0, self.map_size_px[0] - sw)
            cam_y = np.clip(cam_y, 0, self.map_size_px[1] - sh)

            sub = self.background.subsurface((cam_x, cam_y, sw, sh))
            self.screen.blit(sub, (0, 0))

            for obs in self.observers:
                ox, oy = (obs - [cam_x, cam_y]).astype(int)
                pygame.draw.circle(self.screen, (255, 255, 0), (ox, oy), 16)

            ax, ay = (self.agent_pos - [cam_x, cam_y]).astype(int)
            pygame.draw.circle(self.screen, (255, 0, 0), (ax, ay), 16)

            font = pygame.font.SysFont("Arial", 20)
            if logits is None:
                logits = "None"
            lines = [
                f"Step: {self.steps}",
                f"Score: {getattr(self, 'total_reward', 0):.3f}",
                f"Reward: {getattr(self, 'reward', 0):.3f}",
                f"Action: {getattr(self, 'last_action', 'None')}",
                f"logits: {logits}",
            ]

            for i, line in enumerate(lines):
                text_surface = font.render(line, True, (255, 255, 255))
                self.screen.blit(text_surface, (10, 10 + i * 22))

            pygame.display.flip()
            self.clock.tick(16)

            if self.record:
                # 현재 화면 캡처 (RGB)
                frame = pygame.surfarray.array3d(self.screen)
                frame = frame.swapaxes(0, 1)  # (W, H, C) → (H, W, C)
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                self.video_writer.write(frame)

        except:
            pass

    def start_recording(self, filename="record.avi", fps=16):
        self.record = True
        sw, sh = self.camera_size_px
        fourcc = cv2.VideoWriter_fourcc(*"XVID")  # 또는 "mp4v"
        self.video_writer = cv2.VideoWriter(filename, fourcc, fps, (sw, sh))

    def stop_recording(self):
        if self.video_writer:
            self.video_writer.release()
        self.record = False
        self.video_writer = None
        pygame.quit()

    def _get_image_obs(self):
        state = self.get_stacked_state()
        return np.array(state)

    def _get_vector_obs(self):
        return np.array(self.get_direction_one_hot(self.agent_pos))


if __name__ == "__main__":
    env = EnvAvoidObserver(num_observers=5, random_start=True, move_observer=True)
    seed = 1234
    results = []
    for i in range(3):
        obs = env.reset(seed=seed)
        agent_pos = env.agent_pos.copy()
        observers = [o.copy() for o in env.observers]
        observer_targets = [t.copy() for t in env.observer_targets]
        results.append((agent_pos, observers, observer_targets))
        print(f"Run {i+1}:")
        print("  agent_pos:", agent_pos)
        print("  observers:", observers)
        print("  observer_targets:", observer_targets)
        print()
    # 모든 결과가 동일한지 확인
    all_same = all(
        np.allclose(results[0][0], r[0])
        and all(np.allclose(a, b) for a, b in zip(results[0][1], r[1]))
        and all(np.allclose(a, b) for a, b in zip(results[0][2], r[2]))
        for r in results[1:]
    )
    print("Seed 고정 결과 동일 여부:", all_same)

    # env.start_recording("test_run.mp4")

    pygame.init()
    obs = env.reset()

    obs = env.reset()
    done = False
    action = None

    env.render()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_d:
                    action = 0
                elif event.key == pygame.K_c:
                    action = 1
                elif event.key == pygame.K_s:
                    action = 2
                elif event.key == pygame.K_z:
                    action = 3
                elif event.key == pygame.K_a:
                    action = 4
                elif event.key == pygame.K_q:
                    action = 5
                elif event.key == pygame.K_w:
                    action = 6
                elif event.key == pygame.K_e:
                    action = 7

        obs, reward, done, _ = env.step(action)
        env.render()

    env.stop_recording()
    pygame.quit()
