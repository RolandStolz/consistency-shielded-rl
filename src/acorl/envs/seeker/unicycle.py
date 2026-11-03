"""
Seeker with non-holonomic dynamics.
"""

import dataclasses
from typing import ClassVar

import gymnasium as gym
import numpy as np
import pygame

from acorl.envs.definitions import PygameRenderHookFn
from acorl.envs.seeker.base import SeekerEnv, SeekerEnvConfig

"""
TODO: We should change the seeker structure.
    The dynamics should be decoupled from the environment and given per dependency injection.
    Then we don't need to create 4 different seeker envs.
    We only need to create diffferent dynamic models, which are then given to the env.
    Or maybe we just leave it like that now ...
"""


@dataclasses.dataclass
class SeekerUnicycleConfig(SeekerEnvConfig):
    id: str = 'acorl-envs/SeekerUnicycleEnv'
    mass: float = 1.0
    max_acc_force: float = 2.0
    max_brake_force: float = 2.0
    max_yaw_rate: float = np.pi / 2
    goal_reward: float = 10.0
    time_penalty: float = -0.5
    effort_reward_weight: float = -0.1  # should match dt
    approach_reward_weight: float = 3.0

    def __post_init__(self):
        super().__post_init__()
        if self.mass <= 0:
            raise ValueError('weight must be greater than 0.')
        if self.max_acc_force <= 0:
            raise ValueError('acc_force must be greater than 0.')
        if self.max_brake_force <= 0:
            raise ValueError('break_force must be greater than 0.')


class SeekerUnicycleEnv(SeekerEnv):
    metadata: ClassVar = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 15,
    }
    _dt = 0.1

    def __init__(
        self,
        config: SeekerUnicycleConfig | None = None,
        render_hook: PygameRenderHookFn | None = None,
        **kwargs,
    ):
        if render_hook is None and 'render_hook' in kwargs:
            render_hook = kwargs.pop('render_hook')

        if config is None:
            config = SeekerUnicycleConfig()
        config_data = config.model_dump()
        config_data.update(kwargs)
        config = SeekerUnicycleConfig(**config_data)

        if config.dim > 2:
            raise ValueError('Non-holonomic seeker environment currently only supports 2D.')

        super().__init__(
            config=config,
            render_hook=render_hook,
        )

        self._weight = config.mass
        self._max_acc_force = config.max_acc_force
        self._max_break_force = config.max_brake_force
        self._max_yaw_rate = config.max_yaw_rate

        # [acc force / break force, steering angle]
        # self.action_space = gym.spaces.Box(np.array([0, 0, -1]), np.ones(3), dtype=self._dtype)
        self.action_space = gym.spaces.Box(
            low=np.array([-self._max_break_force, -self._max_yaw_rate]),
            high=np.array([self._max_acc_force, self._max_yaw_rate]),
            dtype=self._dtype,
        )

        self._agent_speed = self._dtype(0.0)
        self._agent_heading = self._dtype(0.0)

    def _to_body_frame(self, positions: np.ndarray) -> np.ndarray:
        """Transform absolute positions to agent body frame. Accepts (dim,) or (N, dim)."""
        relative = positions - self._agent_position
        c = np.cos(-self._agent_heading)
        s = np.sin(-self._agent_heading)
        rot = np.array([[c, -s], [s, c]], dtype=self._dtype)
        if relative.ndim == 1:
            return rot @ relative
        return (rot @ relative.T).T

    def _agent_state_obs(self, ego: bool) -> np.ndarray:
        # Heading is implicit (== 0) in the body frame, so the ego segment is speed only.
        if ego:
            return np.array([self._agent_speed], dtype=self._dtype)
        return np.array([self._agent_speed, self._agent_heading], dtype=self._dtype)

    def _agent_state_size(self, ego: bool) -> int:
        return 1 if ego else 2

    def _get_info(self):
        info = super()._get_info()
        if self._use_ego_centric_obs:
            # base stashes agent_position/obstacle_*; add the heading-frame state.
            info['agent_heading'] = float(self._agent_heading)
            info['agent_speed'] = float(self._agent_speed)
        return info

    def _deterministic_initialization(self):
        super()._deterministic_initialization()
        self._agent_speed = self._dtype(0.0)
        self._agent_heading = self._dtype(
            np.arctan2(
                self._goal_position[1] - self._agent_position[1],
                self._goal_position[0] - self._agent_position[0],
            )
        )

    def _random_initialization(self):
        super()._random_initialization()
        self._agent_speed = self._dtype(0.0)

        # random heading
        # self._agent_heading = np.random.uniform(low=-np.pi, high=np.pi)
        self._agent_heading = self._dtype(
            np.arctan2(
                self._goal_position[1] - self._agent_position[1],
                self._goal_position[0] - self._agent_position[0],
            )
        )

    def _dense_reward(
        self, prev_dist_to_goal: float, dist_to_goal: float, action: np.ndarray
    ) -> float:
        effort = max(float(action[0]), 0.0)  # penalize throttle only; braking is free
        approach = prev_dist_to_goal - dist_to_goal
        return self._effort_reward_weight * effort + self._approach_reward_weight * approach  # pyright: ignore[reportReturnType]

    def _agent_dynamics(self, action: np.ndarray, noise: np.ndarray):
        # Semi-implicit Euler integration
        # action: [force_mag (N), yaw_rate (rad/s)]
        # noise:  [d_force (N), d_yaw_rate (rad/s)]  actuator noise
        force_mag, yaw_rate = action
        d_force, d_yaw_rate = noise

        force_mag = np.clip(force_mag + d_force, -self._max_break_force, self._max_acc_force)
        yaw_rate = np.clip(yaw_rate + d_yaw_rate, -self._max_yaw_rate, self._max_yaw_rate)

        # semi-implicit Euler (lets acceleration affect motion within the step)
        longit_acc = force_mag / self._weight
        self._agent_speed = max(0.0, self._agent_speed + longit_acc * self._dt)

        self._agent_heading = self._wrap_angle(self._agent_heading + yaw_rate * self._dt)
        velocity = np.array(
            [
                self._agent_speed * np.cos(self._agent_heading),
                self._agent_speed * np.sin(self._agent_heading),
            ],
            dtype=self._dtype,
        )
        self._agent_position += velocity * self._dt

    def _wrap_angle(self, angle: float) -> np.floating:
        return self._dtype(((angle + np.pi) % (2 * np.pi)) - np.pi)

    def _draw_agent(self, surface, x, y, offset):
        stretch_factor = self.screen_dim / (2 * self._size)
        center = (
            int(
                ((self._agent_position[x] + self._size) * stretch_factor) + offset * self.screen_dim
            ),
            int((self._agent_position[y] + self._size) * stretch_factor),
        )

        pygame.draw.circle(
            surface,
            (0, 0, 255),
            center,
            int(self._agent_radius * stretch_factor),
        )

        heading_vector = np.zeros(self.dim, dtype=self._dtype)
        heading_vector[0] = np.cos(self._agent_heading)
        heading_vector[1] = np.sin(self._agent_heading)

        arrow_length = max(6, int(2.5 * self._agent_radius * stretch_factor))
        arrow_tip = (
            int(center[0] + heading_vector[x] * arrow_length),
            int(center[1] + heading_vector[y] * arrow_length),
        )
        pygame.draw.line(surface, (0, 0, 255), center, arrow_tip, width=2)

    @staticmethod
    def get_attributes_from_obs(
        observation: np.ndarray, config: SeekerEnvConfig
    ) -> tuple[np.ndarray, ...]:
        dim = config.dim

        if isinstance(config, SeekerUnicycleConfig) and config.use_ego_centric_obs:
            agent_speed = observation[0]
            dist_to_boundary = observation[1]
            goal_rel = observation[2 : 2 + dim]

            obstacle_data = observation[2 + dim :]
            num_obstacles = len(obstacle_data) // (dim + 1)
            obstacles_data = obstacle_data.reshape(num_obstacles, dim + 1)
            obstacle_rel_position = obstacles_data[:, :-1]
            obstacle_radius = obstacles_data[:, -1]

            return (
                agent_speed,
                dist_to_boundary,
                goal_rel,
                obstacle_rel_position,
                obstacle_radius,
            )

        agent_position = observation[:dim]
        agent_speed = observation[dim]
        agent_heading = observation[dim + 1]
        goal_position = observation[dim + 2 : 2 * dim + 2]

        obstacle_data = observation[2 * dim + 2 :]
        num_obstacles = len(obstacle_data) // (dim + 1)

        obstacles_data = obstacle_data.reshape(num_obstacles, dim + 1)

        obstacle_position = obstacles_data[:, :-1]
        obstacle_radius = obstacles_data[:, -1]

        return (
            agent_position,
            agent_speed,
            agent_heading,
            goal_position,
            obstacle_position,
            obstacle_radius,
        )
