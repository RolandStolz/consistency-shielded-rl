from typing import Any

import cvxpy as cp
import gymnasium as gym
import numpy as np

from acorl.common.definitions import InputAdaptionCalculator
from acorl.common.exceptions import InputAdaptionComputationError
from acorl.envs.seeker.unicycle import SeekerUnicycleConfig, SeekerUnicycleEnv


class SeekerUnicycleClosestPointProjection(InputAdaptionCalculator[np.ndarray, np.ndarray]):
    """
    Discrete Crontrol-Barrier-Function from https://www.roboticsproceedings.org/rss13/p73.pdf,
    based on first-order taylor approximation of the dynamics.
    """

    def __init__(
        self, env_config: SeekerUnicycleConfig, gamma: float = 0.5, barrier_margin: float = 0.1
    ) -> None:
        """
        The barrier margin controls the under-approximation (comfort distance kept from
        obstacles). Safety no longer relies on it: adapt_input verifies the chosen action
        against the true nonlinear barrier and falls back to the fail-safe if it would be
        unsafe, so the margin can stay small enough to squeeze through tight gaps.
        """
        self._dt = SeekerUnicycleEnv._dt
        self.cfg = env_config
        if not self.cfg.noise == 0.0:
            raise ValueError('CBF-based input adaption does not support actuator noise currently.')

        if not (0.0 < gamma <= 1.0):
            raise ValueError(f'gamma must be in (0, 1], got {gamma}')
        self.gamma = gamma

        if not (0.0 <= barrier_margin):
            raise ValueError(f'barrier_margin must be non-negative, got {barrier_margin}')
        self.barrier_margin = barrier_margin

        self.boundary_size = 10

    def _walls(self):
        # (outward unit normal w, offset b) for the box [-B, B]^2
        return [
            (np.array([1.0, 0.0]), self.boundary_size),
            (np.array([-1.0, 0.0]), self.boundary_size),
            (np.array([0.0, 1.0]), self.boundary_size),
            (np.array([0.0, -1.0]), self.boundary_size),
        ]

    def _barriers(self, x: np.ndarray, obstacles: np.ndarray, radii: np.ndarray) -> np.ndarray:
        """Safe-stopping barrier: dist-based for obstacles, v*v_app/(2*a_max) for walls (angled-approach-correct)."""
        p, v, alpha = x[:2], x[2], x[3]
        ea = np.array([np.cos(alpha), np.sin(alpha)])
        a_max = self.cfg.max_brake_force / self.cfg.mass
        hs = []
        for o, r in zip(obstacles, radii):
            diff = p - o
            dist = np.linalg.norm(diff)
            n = diff / dist
            v_app = max(0.0, -v * (n @ ea))

            # We should add the distance that the agent can travel in one step to the barrier function, since we are using a discrete-time system.
            # hs.append(dist - r - v_app**2 / (2 * a_max))
            hs.append(dist - r - (self._dt * v_app + v_app**2 / (2 * a_max)))
        for w, b in self._walls():
            v_app = max(0.0, v * (w @ ea))
            # hs.append((b - w @ p) - v * v_app / (2 * a_max))
            hs.append((b - w @ p) - (self._dt * v_app + v * v_app / (2 * a_max)))
        return np.array(hs)

    def _predict(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One-step env dynamics F(x, u), semi-implicit Euler, unclipped (env:112-126)."""
        p, v, alpha = x[:2], x[2], x[3]
        force, omega = u[0], u[1]
        v_next = max(0.0, v + self._dt * force / self.cfg.mass)
        alpha_next = alpha + self._dt * omega
        # wrap angle here?
        alpha_next = ((alpha_next + np.pi) % (2 * np.pi)) - np.pi
        p_next = p + self._dt * v_next * np.array([np.cos(alpha_next), np.sin(alpha_next)])
        return np.array([p_next[0], p_next[1], v_next, alpha_next])

    def _dPhi_du(self, x: np.ndarray, u: np.ndarray):
        v, alpha = x[2], x[3]
        s = 1.0 if (v + self._dt * u[0] / self.cfg.mass) > 0 else 0.0
        v_next = max(0.0, v + self._dt * u[0] / self.cfg.mass)
        a_next = alpha + self._dt * u[1]
        ea = np.array([np.cos(a_next), np.sin(a_next)])
        ea_perp = np.array([-np.sin(a_next), np.cos(a_next)])
        Jp = np.zeros((4, 2))
        Jp[0:2, 0] = s * self._dt**2 / self.cfg.mass * ea
        Jp[2, 0] = s * self._dt / self.cfg.mass
        Jp[0:2, 1] = self._dt**2 * v_next * ea_perp
        Jp[3, 1] = self._dt
        return Jp

    def _dh_dx(self, xt: np.ndarray, obstacles: np.ndarray, radii: np.ndarray):
        p, v, alpha = xt[:2], xt[2], xt[3]
        ea = np.array([np.cos(alpha), np.sin(alpha)])
        ea_perp = np.array([-np.sin(alpha), np.cos(alpha)])
        a_max = self.cfg.max_brake_force / self.cfg.mass
        G = np.zeros((len(radii) + 4, 4))
        for i, (o, r) in enumerate(zip(obstacles, radii)):
            diff = p - o
            d = np.linalg.norm(diff)
            n = diff / d
            c = n @ ea
            beta = 1.0 if (-v * c) > 0 else 0.0
            k = beta * v**2 * c / a_max
            dh_dp = n - (k / d) * (ea - c * n)
            dh_dv = -beta * v * c**2 / a_max
            dh_da = -k * (n @ ea_perp)
            # Need to add the discrete time term to the barrier function, since we are using a discrete-time system.
            # dh_dp = n - (k / d) * (ea - c * n) + beta * self._dt * v / d * (ea - c * n)
            # dh_dv = -beta * v * c**2 / a_max + beta * self._dt * c
            # dh_da = -k * (n @ ea_perp) + beta * self._dt * v * (n @ ea_perp)
            G[i] = [dh_dp[0], dh_dp[1], dh_dv, dh_da]

        for i, (w, b) in enumerate(self._walls()):
            cw = w @ ea
            beta = 1.0 if (v * cw) > 0 else 0.0
            G[len(radii) + i] = [
                -w[0],
                -w[1],
                -beta * v * cw / a_max,
                -beta * v**2 * (w @ ea_perp) / (2 * a_max),
                # -beta * v * cw / a_max - beta * self._dt * cw,
                # -beta * v**2 * (w @ ea_perp) / (2 * a_max) - beta * self._dt * v * (w @ ea_perp),
            ]

        return G

    def _jacobian(
        self, x: np.ndarray, u: np.ndarray, obstacles: np.ndarray, radii: np.ndarray
    ):  # replaces the FD loop -> (n_obs,2)
        return self._dh_dx(self._predict(x, u), obstacles, radii) @ self._dPhi_du(x, u)

    def adapt_input(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        action: np.ndarray,
        action_space: gym.spaces.Box,
    ) -> tuple[np.ndarray, bool]:
        if self.cfg.use_ego_centric_obs:
            agent_position = info['agent_position']
            agent_speed = info['agent_speed']
            agent_heading = info['agent_heading']
            obstacle_positions = info['obstacle_position']
            obstacle_radii = info['obstacle_radius']
        else:
            agent_position, agent_speed, agent_heading, _, obstacle_positions, obstacle_radii = (
                SeekerUnicycleEnv.get_attributes_from_obs(observation, self.cfg)
            )
        # boundary_size = info['boundary_size']
        x = np.array([agent_position[0], agent_position[1], agent_speed, agent_heading])

        h0 = self._barriers(x, obstacle_positions, obstacle_radii) - self.barrier_margin
        rhs = (1 - self.gamma) * h0
        next_x_pred = self._predict(x, action)
        h_1_pred = (
            self._barriers(next_x_pred, obstacle_positions, obstacle_radii) - self.barrier_margin
        )
        J = self._jacobian(x, action, obstacle_positions, obstacle_radii)

        u_safe = cp.Variable(2)
        constraints = [
            h_1_pred + J @ (u_safe - action) >= rhs,
            u_safe[0] >= -self.cfg.max_brake_force,  # Fb
            u_safe[0] <= self.cfg.max_acc_force,  # Fa
            u_safe[1] >= -self.cfg.max_yaw_rate,  # omega
            u_safe[1] <= self.cfg.max_yaw_rate,
        ]

        objective = cp.Minimize(cp.sum_squares(u_safe - action))
        problem = cp.Problem(objective, constraints)  # pyright: ignore[reportArgumentType]
        problem.solve()

        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE] or u_safe.value is None:
            raise InputAdaptionComputationError(
                f'Failed to compute safe input. Problem status: {problem.status}'
            )

        # Clip to action space bounds (the env clips too, so we verify the action that is
        # actually applied).
        clipped = np.clip(u_safe.value, action_space.low, action_space.high)

        # Verify the CBF condition on the TRUE (nonlinear) next-step barrier rather than
        # the linearization the QP enforced. The first-order model can overestimate the
        # realized barrier, so a QP-feasible action may still be unsafe. Checking exactly
        # and falling back to the fail-safe keeps safety without a large conservative
        # margin (which would forbid squeezing through tight gaps).
        h_1_true = (
            self._barriers(self._predict(x, clipped), obstacle_positions, obstacle_radii)
            - self.barrier_margin
        )
        if np.any(h_1_true < rhs - 1e-6):
            # Nudge a bit more towards safety
            clipped = clipped + 0.1 * (clipped - action)
            h_1_true = (
                self._barriers(self._predict(x, clipped), obstacle_positions, obstacle_radii)
                - self.barrier_margin
            )
            if np.any(h_1_true < rhs - 1e-6):
                return np.array([-self.cfg.max_brake_force, 0]), True
            # raise InputAdaptionComputationError(
            #     'QP solution violates the true CBF condition; falling back to fail-safe.'
            # )

        was_adapted = not np.allclose(clipped, action, atol=1e-6)
        return clipped, was_adapted

    def compute_fail_safe_input(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        return np.array([-self.cfg.max_brake_force, 0.01])
