import numpy as np
import pygame
import gymnasium as gym
import gymnasium.spaces as spaces
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class RectEntity:
    """Axis-aligned rectangle represented by its center."""

    center: np.ndarray
    size: np.ndarray
    velocity: np.ndarray

    @property
    def half_size(self) -> np.ndarray:
        return self.size / 2.0

    def bounds(self) -> Tuple[float, float, float, float]:
        half = self.half_size
        return (
            float(self.center[0] - half[0]),
            float(self.center[0] + half[0]),
            float(self.center[1] - half[1]),
            float(self.center[1] + half[1]),
        )


@dataclass
class Ball:
    """Circle represented by its center."""

    center: np.ndarray
    radius: float
    velocity: np.ndarray


class EnvPikachuVolleyBall(gym.Env):
    """Simple Pikachu volleyball-like environment with center-based state."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        *,
        render_mode: Optional[str] = None,
        court_size: Tuple[int, int] = (960, 540),
        fps: int = 60,
        max_steps: int = 1800,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        self.court_width, self.court_height = map(float, court_size)
        self.max_steps = int(max_steps)
        self.time_step = 1.0 / float(fps)
        self.metadata["render_fps"] = fps

        # Physics constants (pixels / second space)
        self.player_speed = 360.0
        self.ai_speed = 320.0
        self.jump_speed = 730.0
        self.gravity_player = 1800.0
        self.gravity_ball = 1500.0
        self.ball_restitution = 0.88
        self.net_restitution = 0.35
        self.ground_damping = 0.65
        self.max_ball_speed = 1300.0

        # Geometry (center-based)
        self.player_size = np.array([70.0, 70.0], dtype=np.float32)
        self.opponent_size = np.array([70.0, 70.0], dtype=np.float32)
        self.net_size = np.array([18.0, self.court_height * 0.6], dtype=np.float32)
        self.ball_radius = 22.0

        self.net = RectEntity(
            center=np.array(
                [self.court_width / 2.0, self.court_height - self.net_size[1] / 2.0],
                dtype=np.float32,
            ),
            size=self.net_size.copy(),
            velocity=np.zeros(2, dtype=np.float32),
        )

        self.player = RectEntity(
            center=np.zeros(2, dtype=np.float32),
            size=self.player_size.copy(),
            velocity=np.zeros(2, dtype=np.float32),
        )
        self.opponent = RectEntity(
            center=np.zeros(2, dtype=np.float32),
            size=self.opponent_size.copy(),
            velocity=np.zeros(2, dtype=np.float32),
        )
        self.ball = Ball(
            center=np.zeros(2, dtype=np.float32),
            radius=float(self.ball_radius),
            velocity=np.zeros(2, dtype=np.float32),
        )

        # Bounds for observations
        player_x_max = (
            self.net.center[0] - self.net.half_size[0] - self.player.half_size[0]
        )
        opponent_x_min = (
            self.net.center[0] + self.net.half_size[0] + self.opponent.half_size[0]
        )

        self.action_space = spaces.MultiBinary(3)
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,
                    0.0,
                    -self.player_speed,
                    -self.jump_speed,
                    opponent_x_min,
                    0.0,
                    -self.ai_speed,
                    -self.jump_speed,
                    0.0,
                    0.0,
                    -self.max_ball_speed,
                    -self.max_ball_speed,
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    player_x_max,
                    self.court_height,
                    self.player_speed,
                    self.jump_speed,
                    self.court_width,
                    self.court_height,
                    self.ai_speed,
                    self.jump_speed,
                    self.court_width,
                    self.court_height,
                    self.max_ball_speed,
                    self.max_ball_speed,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self._rng = np.random.default_rng(seed)
        self.steps = 0
        self.ball_grounded = False
        self.ball_ground_side: Optional[str] = None
        self.last_hit = None

        # Rendering helpers
        self.window: Optional[pygame.Surface] = None
        self.canvas: Optional[pygame.Surface] = None
        self.clock: Optional[pygame.time.Clock] = None
        self.background_color = (235, 245, 255)
        self.ground_color = (210, 230, 245)
        self.player_color = (248, 202, 50)
        self.opponent_color = (240, 170, 40)
        self.net_color = (40, 80, 220)
        self.ball_color = (225, 70, 70)

        self.reset()

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.steps = 0
        self.ball_grounded = False
        self.ball_ground_side = None
        self.last_hit = None

        ground_y_player = self.court_height - self.player.half_size[1]
        ground_y_opponent = self.court_height - self.opponent.half_size[1]

        self.player.center[...] = (
            self.court_width * 0.25,
            ground_y_player,
        )
        self.player.velocity[...] = 0.0

        self.opponent.center[...] = (
            self.court_width * 0.75,
            ground_y_opponent,
        )
        self.opponent.velocity[...] = 0.0

        start_x = float(
            self._rng.uniform(self.court_width * 0.55, self.court_width * 0.65)
        )
        self.ball.center[...] = (
            start_x,
            self.court_height * 0.35,
        )
        self.ball.velocity[...] = (
            float(self._rng.uniform(-280.0, -120.0)),
            float(self._rng.uniform(-360.0, -180.0)),
        )

        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        assert self.action_space.contains(action), "Invalid action"

        if self.ball_grounded:
            # Keep returning terminal state if user steps after termination.
            obs = self._get_obs()
            side = self.ball_ground_side or "left"
            reward = 1.0 if side == "right" else -1.0
            return obs, reward, True, False, self._get_info()

        action = np.asarray(action, dtype=np.int8)
        dt = self.time_step

        self._apply_player_control(action, dt)
        self._update_opponent(dt)
        self._update_ball(dt)

        terminated = False
        truncated = False
        reward = 0.0

        if self.ball_grounded:
            terminated = True
            reward = 1.0 if self.ball_ground_side == "right" else -1.0
        else:
            self.steps += 1
            if self.steps >= self.max_steps:
                truncated = True

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        if self.render_mode not in self.metadata["render_modes"]:
            raise ValueError("Unsupported render mode. Use 'human' or 'rgb_array'.")

        surface = self._draw_scene()
        if self.render_mode == "human":
            self._ensure_display()
            assert self.window is not None
            self.window.blit(surface, (0, 0))
            pygame.display.flip()
            if self.clock is not None:
                self.clock.tick(self.metadata["render_fps"])
            return None
        frame = pygame.surfarray.array3d(surface)
        return np.transpose(frame, (1, 0, 2))

    def close(self):
        if self.canvas is not None:
            pygame.display.quit()
            pygame.quit()
            self.canvas = None
            self.window = None
            self.clock = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.player.center[0],
                self.player.center[1],
                self.player.velocity[0],
                self.player.velocity[1],
                self.opponent.center[0],
                self.opponent.center[1],
                self.opponent.velocity[0],
                self.opponent.velocity[1],
                self.ball.center[0],
                self.ball.center[1],
                self.ball.velocity[0],
                self.ball.velocity[1],
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        return {
            "player_center": tuple(float(v) for v in self.player.center),
            "opponent_center": tuple(float(v) for v in self.opponent.center),
            "ball_center": tuple(float(v) for v in self.ball.center),
            "ball_ground_side": self.ball_ground_side,
            "last_hit": self.last_hit,
        }

    # -- Controls ------------------------------------------------------
    def _apply_player_control(self, action: np.ndarray, dt: float) -> None:
        left, right, jump = action
        horizontal = float(right - left)
        self.player.velocity[0] = horizontal * self.player_speed

        if self._is_on_ground(self.player):
            self.player.center[1] = self.court_height - self.player.half_size[1]
            if jump:
                self.player.velocity[1] = -self.jump_speed
            else:
                self.player.velocity[1] = 0.0
        else:
            self.player.velocity[1] += self.gravity_player * dt

        self.player.center += self.player.velocity * dt
        self._constrain_player()

    def _update_opponent(self, dt: float) -> None:
        target_x = max(
            self.net.center[0]
            + self.net.half_size[0]
            + self.opponent.half_size[0]
            + 12.0,
            min(self.ball.center[0], self.court_width - self.opponent.half_size[0]),
        )
        delta_x = target_x - self.opponent.center[0]
        if abs(delta_x) < 5.0:
            self.opponent.velocity[0] = 0.0
        else:
            direction = np.sign(delta_x)
            self.opponent.velocity[0] = direction * self.ai_speed

        if self._is_on_ground(self.opponent):
            self.opponent.center[1] = self.court_height - self.opponent.half_size[1]
            wants_jump = (
                self.ball.center[0] > self.net.center[0]
                and self.ball.velocity[1] > -10.0
                and self.ball.center[1] < self.court_height * 0.6
                and abs(self.ball.center[0] - self.opponent.center[0])
                < self.opponent.size[0]
            )
            if wants_jump:
                self.opponent.velocity[1] = -self.jump_speed * 0.9
            else:
                self.opponent.velocity[1] = 0.0
        else:
            self.opponent.velocity[1] += self.gravity_player * dt

        self.opponent.center += self.opponent.velocity * dt
        self._constrain_opponent()

    def _update_ball(self, dt: float) -> None:
        if self.ball_grounded:
            return

        self.ball.velocity[1] += self.gravity_ball * dt
        speed = np.linalg.norm(self.ball.velocity)
        if speed > self.max_ball_speed:
            self.ball.velocity *= self.max_ball_speed / max(speed, 1e-6)

        self.ball.center += self.ball.velocity * dt

        # Collisions with world bounds
        if self.ball.center[0] - self.ball.radius < 0.0:
            self.ball.center[0] = self.ball.radius
            self.ball.velocity[0] = abs(self.ball.velocity[0])
        if self.ball.center[0] + self.ball.radius > self.court_width:
            self.ball.center[0] = self.court_width - self.ball.radius
            self.ball.velocity[0] = -abs(self.ball.velocity[0])
        if self.ball.center[1] - self.ball.radius < 0.0:
            self.ball.center[1] = self.ball.radius
            self.ball.velocity[1] = abs(self.ball.velocity[1])

        if self.ball.center[1] + self.ball.radius >= self.court_height:
            self.ball.center[1] = self.court_height - self.ball.radius
            self.ball.velocity[...] = 0.0
            self.ball_grounded = True
            self.ball_ground_side = (
                "left" if self.ball.center[0] < self.net.center[0] else "right"
            )
            return

        # Collisions with net and characters
        collided_net = self._resolve_circle_rect_collision(
            self.ball, self.net, self.net_restitution
        )
        collided_player = self._resolve_circle_rect_collision(
            self.ball, self.player, self.ball_restitution
        )
        collided_opponent = self._resolve_circle_rect_collision(
            self.ball, self.opponent, self.ball_restitution
        )

        if collided_player:
            self.last_hit = "player"
        elif collided_opponent:
            self.last_hit = "opponent"

    # -- Geometry helpers ---------------------------------------------
    def _constrain_player(self) -> None:
        half_w = self.player.half_size[0]
        left_bound = half_w
        right_bound = self.net.center[0] - self.net.half_size[0] - half_w
        new_x = np.clip(self.player.center[0], left_bound, right_bound)
        if new_x != self.player.center[0]:
            self.player.center[0] = new_x
            self.player.velocity[0] = 0.0
        self._resolve_vertical_limits(self.player)

    def _constrain_opponent(self) -> None:
        half_w = self.opponent.half_size[0]
        left_bound = self.net.center[0] + self.net.half_size[0] + half_w
        right_bound = self.court_width - half_w
        new_x = np.clip(self.opponent.center[0], left_bound, right_bound)
        if new_x != self.opponent.center[0]:
            self.opponent.center[0] = new_x
            self.opponent.velocity[0] = 0.0
        self._resolve_vertical_limits(self.opponent)

    def _resolve_vertical_limits(self, entity: RectEntity) -> None:
        half_h = entity.half_size[1]
        top_bound = half_h
        bottom_bound = self.court_height - half_h
        new_y = np.clip(entity.center[1], top_bound, bottom_bound)
        if new_y != entity.center[1] and entity.center[1] > new_y:
            entity.velocity[1] *= self.ground_damping
        entity.center[1] = new_y
        if entity.center[1] >= bottom_bound and entity.velocity[1] > 0.0:
            entity.velocity[1] = 0.0

    def _is_on_ground(self, entity: RectEntity) -> bool:
        return abs(entity.center[1] + entity.half_size[1] - self.court_height) < 1e-2

    def _resolve_circle_rect_collision(
        self,
        ball: Ball,
        rect: RectEntity,
        restitution: float,
    ) -> bool:
        left, right, top, bottom = rect.bounds()
        closest_x = np.clip(ball.center[0], left, right)
        closest_y = np.clip(ball.center[1], top, bottom)
        diff = ball.center - np.array([closest_x, closest_y], dtype=np.float32)
        dist_sq = diff.dot(diff)
        radius_sq = ball.radius * ball.radius

        if dist_sq > radius_sq:
            return False

        distance = float(np.sqrt(max(dist_sq, 1e-9)))
        normal = (
            diff / distance
            if distance > 1e-6
            else np.array([0.0, -1.0], dtype=np.float32)
        )
        penetration = ball.radius - distance
        ball.center += normal * penetration

        relative_velocity = ball.velocity - rect.velocity
        vel_along_normal = float(relative_velocity.dot(normal))
        if vel_along_normal > 0.0:
            return True

        bounce = -(1.0 + restitution) * vel_along_normal
        ball.velocity += normal * bounce
        ball.velocity += rect.velocity * 0.15
        return True

    # -- Rendering -----------------------------------------------------
    def _ensure_display(self) -> None:
        if self.canvas is None:
            pygame.init()
            size = (int(self.court_width), int(self.court_height))
            self.canvas = pygame.Surface(size)
        if self.window is None:
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (int(self.court_width), int(self.court_height))
            )
        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _draw_scene(self) -> pygame.Surface:
        if self.canvas is None:
            pygame.init()
            size = (int(self.court_width), int(self.court_height))
            self.canvas = pygame.Surface(size)
        surface = self.canvas
        surface.fill(self.background_color)
        pygame.draw.rect(
            surface,
            self.ground_color,
            pygame.Rect(
                0,
                int(self.court_height * 0.92),
                int(self.court_width),
                int(self.court_height * 0.08),
            ),
        )

        # Net (blue rectangle)
        net_rect = pygame.Rect(0, 0, int(self.net.size[0]), int(self.net.size[1]))
        net_rect.center = (int(self.net.center[0]), int(self.net.center[1]))
        pygame.draw.rect(surface, self.net_color, net_rect)

        # Player (yellow rectangle)
        player_rect = pygame.Rect(
            0,
            0,
            int(self.player.size[0]),
            int(self.player.size[1]),
        )
        player_rect.center = (int(self.player.center[0]), int(self.player.center[1]))
        pygame.draw.rect(surface, self.player_color, player_rect, border_radius=10)

        # Opponent (darker yellow/orange)
        opponent_rect = pygame.Rect(
            0,
            0,
            int(self.opponent.size[0]),
            int(self.opponent.size[1]),
        )
        opponent_rect.center = (
            int(self.opponent.center[0]),
            int(self.opponent.center[1]),
        )
        pygame.draw.rect(surface, self.opponent_color, opponent_rect, border_radius=10)

        # Ball (red circle)
        pygame.draw.circle(
            surface,
            self.ball_color,
            (int(self.ball.center[0]), int(self.ball.center[1])),
            int(self.ball.radius),
        )

        return surface


if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="Quick smoke test for EnvPikachuBeachBall"
    )
    parser.add_argument(
        "--steps", type=int, default=6000, help="Number of steps to simulate"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Seed for random actions"
    )
    parser.add_argument(
        "--render-mode",
        choices=["none", "human", "rgb_array"],
        default="human",
        help="Rendering mode to use during the test run",
    )
    args = parser.parse_args()

    render_mode = None if args.render_mode == "none" else args.render_mode
    env = EnvPikachuVolleyBall(render_mode=render_mode)

    while True:
        obs, info = env.reset(seed=args.seed)
        print("Initial observation:", obs)
        print("Initial info:", info)

        rng = np.random.default_rng(args.seed)
        try:
            for step in range(args.steps):
                action = rng.integers(0, 2, size=3, dtype=np.int8)
                obs, reward, terminated, truncated, info = env.step(action)
                print(
                    f"step={step:04d} action={action.tolist()} reward={reward:.2f} "
                    f"terminated={terminated} truncated={truncated}"
                )
                if render_mode:
                    frame = env.render()
                    if render_mode == "human":
                        pygame.event.pump()
                    elif render_mode == "rgb_array":
                        print(f"Frame shape: {None if frame is None else frame.shape}")
                if terminated or truncated:
                    print("Episode finished", info)
                    break
                time.sleep(0.01 if render_mode == "human" else 0.0)
        except KeyboardInterrupt:
            print("Interrupted by user")
            break
        finally:
            env.close()
