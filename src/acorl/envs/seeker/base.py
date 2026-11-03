import dataclasses
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import pygame
from gymnasium.core import RenderFrame

from acorl.common.definitions import BaseEnvConfig
from acorl.envs.definitions import PygameRenderHookFn
from acorl.envs.geometry import any_circle_overlap, circle_contains_point


@dataclasses.dataclass
class SeekerEnvConfig(BaseEnvConfig):
    id: str = 'acorl-envs/SeekerEnv'
    randomize: bool = True
    num_obstacles: int = 3
    dim: int = 2
    noise: float = 0.0
    action_size: float = 1.0
    use_dense_reward: bool = True
    done_on_collision: bool = True
    done_on_goal: bool = True
    collision_reward: float = -20.0
    goal_reward: float = 10.0
    time_penalty: float = -0.3
    approach_reward_weight: float = 5.0
    effort_reward_weight: float = -0.1

    use_ego_centric_obs: bool = False

    render_axis: set | None = None
    log: bool = False

    def __post_init__(self):
        if self.dim < 2:
            raise ValueError('dim must be greater than or equal to 2.')
        if self.noise is not None and self.noise < 0:
            raise ValueError('noise must be positive or None.')


# TODO: Think about maybe setting self._size = 1, to have standardized rewards by default
class SeekerEnv(gym.Env[np.ndarray, np.ndarray]):
    screen = None
    screen_dim = 512
    clock = None
    _size = 10
    _dtype = np.float64
    _initialization_retries = 1000
    _dt = 1

    metadata: ClassVar = {  # pyright: ignore[reportIncompatibleVariableOverride]
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 5,
    }

    def __init__(
        self,
        config: SeekerEnvConfig | None = None,
        render_hook: PygameRenderHookFn | None = None,
        **kwargs: Any,
    ):
        if config is None:
            config = SeekerEnvConfig()
        if kwargs:
            config_data = config.model_dump()
            config_data.update(kwargs)
            config = type(config)(**config_data)
        self.config = config

        if config.render_axis is None:
            render_axis = set()
            render_axis.add(frozenset({0, 1}))
        self.dim = config.dim
        self._num_obstacles = config.num_obstacles
        self.randomize = config.randomize
        self.remaining_retries = self._initialization_retries
        self.render_mode = config.render_mode
        self.noise = config.noise
        self.render_hook = render_hook
        self.render_axis = render_axis  # pyright: ignore[reportPossiblyUnboundVariable]
        self.log = config.log
        self._use_ego_centric_obs = config.use_ego_centric_obs

        self._collision_reward = config.collision_reward
        self._goal_reward = config.goal_reward
        # The per-step shaping terms are rates: the config expresses them per unit of
        # simulated time (dt=1 base), so scale by the integration timestep. A smaller
        # dt means more steps cover the same simulated time, so the per-step cost/gain
        # must shrink to keep the trajectory-integrated reward invariant to dt. The
        # terminal goal/collision rewards are one-off events, not rates, so unscaled.
        self._time_penalty = config.time_penalty * self._dt
        self._approach_reward_weight = config.approach_reward_weight * self._dt
        self._effort_reward_weight = config.effort_reward_weight * self._dt
        self._use_dense_reward = config.use_dense_reward
        self._done_on_collision = config.done_on_collision
        self._done_on_goal = config.done_on_goal

        if self._use_ego_centric_obs:
            # [<agent_state>, dist_to_boundary, goal_rel(dim), (obs_rel(dim) + radius) * N]
            obs_shape = (
                self._agent_state_size(ego=True)
                + 1
                + self.dim
                + self._num_obstacles * (self.dim + 1)
            )
            obs_bound = 2 * self._size
        else:
            obs_shape = (
                self.dim
                + self._agent_state_size(ego=False)
                + self.dim
                + self._num_obstacles * (self.dim + 1)
            )
            obs_bound = self._size
        self.observation_space = gym.spaces.Box(
            low=-obs_bound, high=obs_bound, shape=(obs_shape,), dtype=self._dtype
        )

        self.action_size = config.action_size
        self.action_space = gym.spaces.Box(
            low=-self.action_size, high=self.action_size, shape=(self.dim,), dtype=self._dtype
        )

        # Observations
        self._agent_position = np.zeros(self.dim, dtype=self._dtype)
        self._agent_radius = 0.02 * self._size
        self._obstacle_position = np.zeros((self._num_obstacles, self.dim), dtype=self._dtype)
        self._obstacle_radius = np.zeros(self._num_obstacles, dtype=self._dtype)
        self._goal_position = np.zeros(self.dim, dtype=self._dtype)
        self._goal_radius = 0.1 * self._size

        if self._num_obstacles != 1 and not self.randomize and self._num_obstacles != 3:
            self._num_obstacles = 3
            gym.logger.warn('Using 3 obstacles for deterministic initialization')

    def _agent_state_obs(self, ego: bool) -> np.ndarray:
        """Env-specific agent internal state (e.g. velocity, speed). Empty for base."""
        return np.array([], dtype=self._dtype)

    def _agent_state_size(self, ego: bool) -> int:
        """Length of the segment returned by `_agent_state_obs`. Used for obs shape."""
        return 0

    def _to_body_frame(self, positions: np.ndarray) -> np.ndarray:
        """Transform absolute positions to the agent frame. Accepts (dim,) or (N, dim).

        The base (holonomic) frame is translation-only; subclasses with a heading
        override this to add rotation.
        """
        return positions - self._agent_position

    def _ego_world_segment(self) -> np.ndarray:
        """Shared ego-centric world description: boundary distance + relative entities."""
        dist_to_boundary = self._dtype(np.min(self._size - np.abs(self._agent_position)))
        goal_rel = self._to_body_frame(self._goal_position)
        obs_rel = self._to_body_frame(self._obstacle_position)
        obstacles_data = np.column_stack((obs_rel, self._obstacle_radius))
        return np.concatenate(
            [np.array([dist_to_boundary], dtype=self._dtype), goal_rel, obstacles_data.flatten()]
        )

    def _get_obs(self) -> np.ndarray:
        if self._use_ego_centric_obs:
            return np.concatenate([self._agent_state_obs(ego=True), self._ego_world_segment()])

        obstacles_data = np.column_stack((self._obstacle_position, self._obstacle_radius))
        return np.concatenate(
            [
                self._agent_position,
                self._agent_state_obs(ego=False),
                self._goal_position,
                obstacles_data.flatten(),
            ]
        )

    def _get_info(self):
        info = {
            'distance': np.linalg.norm(self._goal_position - self._agent_position),
            'boundary_size': self._size,
            'collision': False,
            'goal_reached': False,
        }
        if self._use_ego_centric_obs:
            # Stash absolute world-frame state so downstream consumers (constraint
            # calculators, render hooks) can recover it from the ego-centric obs.
            info['agent_position'] = self._agent_position.copy()
            info['obstacle_position'] = self._obstacle_position.copy()
            info['obstacle_radius'] = self._obstacle_radius.copy()
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)
        self.remaining_retries = self._initialization_retries

        if self.randomize:
            self._random_initialization()
        else:
            self._deterministic_initialization()

        return self._get_obs(), self._get_info()

    def _deterministic_initialization(self):
        self._agent_position = (
            np.zeros(self.dim, dtype=self._dtype)
            - 6
            + np.random.uniform(-0.1, 0.1, self.dim).astype(self._dtype)
        )
        self._goal_position = np.zeros(self.dim, dtype=self._dtype) + 6

        if self._num_obstacles == 1:
            self._obstacle_position = np.zeros((1, self.dim), dtype=self._dtype)
            self._obstacle_radius = np.array([4], dtype=self._dtype)
        else:
            assert self._num_obstacles == 3, (
                'Only 1 or 3 obstacles are supported in deterministic initialization'
            )
            self._obstacle_position = np.zeros((self._num_obstacles, self.dim), dtype=self._dtype)
            self._obstacle_position[1][0] -= 7
            self._obstacle_position[1][1] += 4
            self._obstacle_position[2][0] += 4
            self._obstacle_position[2][1] -= 7

            self._obstacle_radius = np.array([4, 2, 2], dtype=self._dtype)

        assert not self._check_collision(), 'Deterministic initialization results in collision!'

    def _random_init_single(self, with_obstacle: bool = True):
        # TODO: this can be implemented more elegantly
        self._agent_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))

        # Sample the goal position from the other half of the box, where the agent is not
        self._goal_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))
        while np.linalg.norm(self._agent_position - self._goal_position) < self._size / 2:
            self._goal_position = np.random.uniform(-self._size + 1, self._size - 1, (self.dim,))

        if with_obstacle:
            # Get a random point on the line between agent and goal
            obstacle_pos = self._agent_position + np.random.uniform(0.4, 0.6) * (
                self._goal_position - self._agent_position
            )

            min_radius = 1
            max_radius = min(
                np.linalg.norm(self._agent_position - self._goal_position) / 3,
                np.max(np.concatenate((obstacle_pos - self._size, obstacle_pos + self._size))),
            )

            radius = float(np.random.uniform(min_radius, max_radius))

            if circle_contains_point(
                obstacle_pos, radius, self._agent_position
            ) or circle_contains_point(obstacle_pos, radius, self._goal_position):
                if self.log:
                    print('Repeating random initialization!')
                self._random_initialization()

            # Construct the obstacle
            self._obstacle_position = np.array([obstacle_pos])
            self._obstacle_radius = np.array([radius])

    def _random_init_multiple_many(self):
        # use single implementation to ensure there is an obstacle between agent and goal
        self._random_init_single(with_obstacle=False)

        # # Sobol sampling for pseudo-uniform (more grid like) samples
        # from scipy.stats import qmc
        # sampler = qmc.Sobol(
        #     d=self.dim, scramble=False
        # )  # scramble=False for a deterministic sequence
        # samples = sampler.random(n=self._num_obstacles)
        # obstacle_positions = qmc.scale(samples, -self._size, self._size)

        # Uniform sampling
        obstacle_positions = np.random.uniform(
            -self._size, self._size, size=(self._num_obstacles, self.dim)
        )

        # stack with agent and goal positions
        entity_positions = np.vstack(
            (obstacle_positions, self._agent_position, self._goal_position)
        )

        # Loop-based
        # max_radii = np.zeros(self._num_obstacles)
        # for i in range(self._num_obstacles):
        #     mask_not_i = np.arange(self._num_obstacles + 2) != i
        #     max_radius = (
        #         np.min(np.linalg.norm(entity_positions[mask_not_i] - obstacle_positions[i], axis=1))
        #         * 0.9
        #     )
        #     max_radii[i] = max_radius

        # Vectorized computation
        diffs = obstacle_positions[:, np.newaxis, :] - entity_positions[np.newaxis, :, :]
        all_distances = np.linalg.norm(diffs, axis=2)

        diag_indices = np.arange(self._num_obstacles)
        all_distances[diag_indices, diag_indices] = np.inf

        min_other_dist = np.min(all_distances, axis=1)
        max_radii = min_other_dist * 0.9

        # max_radii[max_radii < minimal_max_radius] = minimal_max_radius
        # clamp: low > high makes np.random.uniform exceed max_radii, spawning agent/goal inside
        min_radius = np.minimum(self._agent_radius, max_radii)

        obstacle_radii = np.random.uniform(min_radius, max_radii)

        self._obstacle_position = obstacle_positions
        self._obstacle_radius = obstacle_radii

    def _random_init_multiple_few(self):
        # use single implementation to ensure there is an obstacle between agent and goal
        self._random_init_single()

        # add remaining obstacles at random, ensuring that none overlap to form a barrier for the agent
        entity_positions = [self._agent_position, self._goal_position, self._obstacle_position[0]]
        entity_radii = [self._agent_radius, self._goal_radius, self._obstacle_radius[0]]

        # max_radius = self._size / math.sqrt(self._size)
        max_radius = (2.0 * self._size) / self._num_obstacles
        min_radius = self._agent_radius
        for _ in range(self._num_obstacles - 1):
            # Create a random obstacles parameters
            temp_obstacle_pos = np.random.uniform(-self._size, self._size, (self.dim,))
            temp_obstacle_radius = np.random.uniform(min_radius, max_radius)
            # Check for overlaps (safety distance to make sure that the path to the goal is not blocked)
            while any_circle_overlap(
                temp_obstacle_pos,
                temp_obstacle_radius,
                entity_positions,
                entity_radii,
                safety_distance=2 * self._agent_radius + 1,
            ):
                self.remaining_retries -= 1
                temp_obstacle_pos = np.random.uniform(-self._size, self._size, (self.dim,))
                temp_obstacle_radius = np.random.uniform(min_radius, max_radius)
                if self.remaining_retries <= 0:
                    break

            if self.remaining_retries <= 0:
                raise RuntimeError(
                    'You have run out of retries to find valid obstacle positions.\n'
                    f'random_retries = {self._initialization_retries}; '
                    f'num_obstacles = {self._num_obstacles}; '
                    f'obstacles inserted = {len(self._obstacle_position)}'
                )

            entity_positions += [temp_obstacle_pos]
            entity_radii += [temp_obstacle_radius]

            self._obstacle_position = np.concatenate((self._obstacle_position, [temp_obstacle_pos]))
            self._obstacle_radius = np.concatenate((self._obstacle_radius, [temp_obstacle_radius]))

        self.remaining_retries = self._initialization_retries
        self._num_obstacles = len(self._obstacle_position)

    def _shuffle_obstacles(self):
        # The on-path obstacle is always constructed first; shuffle so it does not always
        # occupy observation slot 0 (avoids the policy keying on that positional artifact).
        n = len(self._obstacle_position)
        if n <= 1:
            return
        perm = np.random.permutation(n)
        self._obstacle_position = self._obstacle_position[perm]
        self._obstacle_radius = self._obstacle_radius[perm]

    def _random_initialization(self):
        if self._num_obstacles == 1:
            self._random_init_single()
        elif self._num_obstacles <= self.dim * 4:
            self._random_init_multiple_few()
        else:
            self._random_init_multiple_many()

        self._shuffle_obstacles()

    def _agent_dynamics(self, action: np.ndarray, noise: np.ndarray):
        self._agent_position = self._agent_position + action + noise

    def _dense_reward(
        self, prev_dist_to_goal: float, dist_to_goal: float, action: np.ndarray
    ) -> float:
        effort = np.linalg.norm(action) ** 2
        approach = prev_dist_to_goal - dist_to_goal
        return self._effort_reward_weight * effort + self._approach_reward_weight * approach  # pyright: ignore[reportReturnType]

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        prev_pos = self._agent_position.copy()

        sampled_noise = np.zeros(self.dim)
        if self.noise is not None:
            sampled_noise = np.random.uniform(low=-self.noise, high=self.noise, size=self.dim)

        self._agent_dynamics(action, sampled_noise)

        reward = self._time_penalty
        if self._use_dense_reward:
            prev_dist_to_goal = np.linalg.norm(prev_pos - self._goal_position).item()
            dist_to_goal = np.linalg.norm(self._goal_position - self._agent_position).item()
            reward += self._dense_reward(prev_dist_to_goal, dist_to_goal, action)

        terminated = False
        truncated = False

        info = self._get_info()

        if self._check_collision():
            reward = self._collision_reward
            terminated = True and self._done_on_collision
            info['collision'] = True
            if self.log:
                print('Collision detected!')

        if self._goal_reached():
            reward = self._goal_reward
            terminated = True and self._done_on_goal
            info['goal_reached'] = True
            if self.log:
                print('Goal reached!')

        return self._get_obs(), reward, terminated, truncated, info

    def _goal_reached(self) -> bool:
        # Agent is treated as a point (its radius is visualization-only)
        return bool(np.linalg.norm(self._goal_position - self._agent_position) < self._goal_radius)

    def _check_collision(self) -> bool:
        if np.any(np.abs(self._agent_position) > self._size):
            return True

        for i in range(self._num_obstacles):
            if circle_contains_point(
                self._obstacle_position[i], self._obstacle_radius[i].item(), self._agent_position
            ):
                return True

        return False

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                'You are calling render method without specifying any render mode. '
                'You can specify the render_mode at initialization, '
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        return self._render_frame()  # type: ignore

    def _draw_agent(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        pygame.draw.circle(
            surface,
            (0, 0, 255),
            (
                int(
                    ((self._agent_position[x] + self._size) * stretch_factor)
                    + offset * self.screen_dim
                ),
                int((self._agent_position[y] + self._size) * stretch_factor),
            ),
            int(self._agent_radius * stretch_factor),
        )

    def _draw_obstacle(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        for i in range(self._num_obstacles):
            pygame.draw.circle(
                surface,
                (255, 0, 0),
                (
                    int(
                        ((self._obstacle_position[i][x] + self._size) * stretch_factor)
                        + offset * self.screen_dim
                    ),
                    int((self._obstacle_position[i][y] + self._size) * stretch_factor),
                ),
                int(self._obstacle_radius[i] * stretch_factor),
            )

    def _draw_goal(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        pygame.draw.circle(
            surface,
            (0, 255, 0),
            (
                int(
                    ((self._goal_position[x] + self._size) * stretch_factor)
                    + offset * self.screen_dim
                ),
                int((self._goal_position[y] + self._size) * stretch_factor),
            ),
            int(self._goal_radius * stretch_factor),
        )

    def _draw_label(self, surface, x, y, offset):
        font = pygame.font.Font(None, 30)
        text_surface = font.render(f'x: dim {x}   y: dim {y}', True, (0, 0, 0))
        surface.blit(text_surface, (int(offset * self.screen_dim), 0))

    def _render_frame(self) -> RenderFrame | list[RenderFrame] | np.ndarray | None:
        num_displays = len(self.render_axis)
        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((self.screen_dim * num_displays, self.screen_dim))
            pygame.display.set_caption('Seeker')

        if self.clock is None:
            self.clock = pygame.time.Clock()

        surface = pygame.Surface((self.screen_dim * num_displays, self.screen_dim))
        surface.fill((255, 255, 255))

        for offset, view in enumerate(self.render_axis):
            ax_iter = iter(view)
            x, y = next(ax_iter), next(ax_iter)

            self._draw_obstacle(surface, x, y, offset)

            # This allows to insert any rendering from outside the class (e.g. rendering a safe
            # action set)
            if self.render_hook is not None:
                info = self._get_info()
                info['screen_size'] = self.screen_dim
                self.render_hook(self._get_obs(), info, surface=surface)

            self._draw_agent(surface, x, y, offset)
            self._draw_goal(surface, x, y, offset)
            self._draw_label(surface, x, y, offset)

        line_color = (0, 0, 0)
        line_width = 1

        for i in range(1, num_displays):
            line_x = i * self.screen_dim  # Position of the dividing line
            pygame.draw.line(
                surface, line_color, (line_x, 0), (line_x, self.screen_dim), line_width
            )

        self.screen.blit(surface, surface.get_rect())
        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.metadata['render_fps'])
            pygame.display.update()
            return None

        else:  # rgb_array
            return np.transpose(np.array(pygame.surfarray.pixels3d(surface)), axes=(1, 0, 2))

    def set_render_hook(self, render_hook: PygameRenderHookFn):
        self.render_hook = render_hook

    def close(self):
        pygame.display.quit()
        pygame.quit()

    @staticmethod
    def get_attributes_from_obs(
        observation: np.ndarray, config: SeekerEnvConfig
    ) -> tuple[np.ndarray, ...]:
        dim = config.dim
        num_obstacles = config.num_obstacles

        if config.use_ego_centric_obs:
            # [dist_to_boundary, goal_rel(dim), (obs_rel(dim) + radius) * N]
            dist_to_boundary = observation[0]
            goal_rel = observation[1 : 1 + dim]
            obstacle_data = observation[1 + dim : 1 + dim + num_obstacles * (dim + 1)]
            obstacles_data = obstacle_data.reshape(num_obstacles, dim + 1)
            return dist_to_boundary, goal_rel, obstacles_data[:, :-1], obstacles_data[:, -1]

        agent_position = observation[:dim]
        goal_position = observation[dim : 2 * dim]

        obstacle_data = observation[2 * dim : 2 * dim + num_obstacles * (dim + 1)]
        num_obstacles = len(obstacle_data) // (dim + 1)

        obstacles_data = obstacle_data.reshape(num_obstacles, dim + 1)

        obstacle_position = obstacles_data[:, :-1]
        obstacle_radius = obstacles_data[:, -1]

        return agent_position, goal_position, obstacle_position, obstacle_radius
