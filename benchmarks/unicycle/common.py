"""
Shared helpers for the unicycle closest-point-projection experiments.

The safety shield (`SeekerUnicycleClosestPointProjection`) reads the dynamics parameters
(`mass`, `max_acc_force`, `max_brake_force`, `max_yaw_rate`) from the config it is built with.
That config is decoupled from the config the env is registered with, so a shield can guard an
env whose true dynamics differ from what the shield assumes -- this is what the V-SUM
consistency experiments exploit (shield adapted vs. not adapted).
"""

import os

import gymnasium as gym
import numpy as np
import torch
from tianshou.highlevel.config import OnPolicyTrainingConfig
from tianshou.highlevel.env import EnvMode, VectorEnvType
from tianshou.highlevel.experiment import ExperimentConfig, PPOExperimentBuilder
from tianshou.highlevel.logger import LoggerFactory
from tianshou.highlevel.params.algorithm_params import PPOParams
from tianshou.highlevel.persistence import PolicyPersistence

from acorl.envs.constraints.seeker.unicycle import SeekerUnicycleClosestPointProjection
from acorl.envs.seeker.unicycle import SeekerUnicycleConfig
from acorl.tianshou.highlevel.env_factory import InputAdaptionEnvFactory

WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'unicycle-paper')
WANDB_ENTITY = os.getenv('WANDB_ENTITY')

SHIELD_GAMMA = 0.5
NUM_OBSTACLES = 10

# Consistent ("nominal") dynamics the base agents are trained on: F_a = F_b = 2, m = 1.
BASELINE = {'mass': 1.0, 'max_acc_force': 2.0, 'max_brake_force': 2.0}
DYNAMICS_KEYS = ('mass', 'max_acc_force', 'max_brake_force')

# Perturbation grid (one parameter off baseline at a time).
F_RANGE = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
M_RANGE = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]


def build_grid() -> list[dict]:
    """One-parameter-at-a-time perturbation grid; the shared baseline (2,2,1) appears once.

    Each entry has `param_varied` plus the three dynamics keys."""
    grid = [{'param_varied': 'baseline', **BASELINE}]
    for v in F_RANGE:
        if v != BASELINE['max_acc_force']:
            grid.append({'param_varied': 'max_acc_force', **BASELINE, 'max_acc_force': v})
    for v in F_RANGE:
        if v != BASELINE['max_brake_force']:
            grid.append({'param_varied': 'max_brake_force', **BASELINE, 'max_brake_force': v})
    for v in M_RANGE:
        if v != BASELINE['mass']:
            grid.append({'param_varied': 'mass', **BASELINE, 'mass': v})
    return grid


def config_tag(cfg: dict) -> str:
    """Stable identifier for a grid config, e.g. 'baseline', 'mass_1.2', 'max_brake_force_0.5'."""
    if cfg['param_varied'] == 'baseline':
        return 'baseline'
    p = cfg['param_varied']
    return f'{p}_{cfg[p]}'


# PPO algo fields set on PPOParams from a swept/exported config (the rest are training-config fields).
PPO_PARAM_FIELDS = ('lr', 'gamma', 'gae_lambda', 'ent_coef', 'vf_coef', 'eps_clip', 'max_grad_norm')


def make_env_config(
    mass: float = BASELINE['mass'],
    max_acc_force: float = BASELINE['max_acc_force'],
    max_brake_force: float = BASELINE['max_brake_force'],
) -> SeekerUnicycleConfig:
    return SeekerUnicycleConfig(
        num_obstacles=NUM_OBSTACLES,
        use_ego_centric_obs=True,
        use_dense_reward=True,
        mass=mass,
        max_acc_force=max_acc_force,
        max_brake_force=max_brake_force,
        goal_reward=10.0,
        approach_reward_weight=3.0,
    )


# Normalized throttle command seen by the policy: u[0] in [-1, 1] is the fraction of max_brake_force
# (u < 0) or max_acc_force (u >= 0), u[1] the fraction of max_yaw_rate.
THROTTLE_SPACE = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)


def to_force(u: np.ndarray, cfg: SeekerUnicycleConfig) -> np.ndarray:
    """Throttle command -> [force (N), yaw rate (rad/s)] under the actuator gains of `cfg`."""
    gain = cfg.max_acc_force if u[0] >= 0 else cfg.max_brake_force
    return np.array([u[0] * gain, u[1] * cfg.max_yaw_rate])


def to_throttle(a: np.ndarray, cfg: SeekerUnicycleConfig) -> np.ndarray:
    """Inverse of `to_force`."""
    gain = cfg.max_acc_force if a[0] >= 0 else cfg.max_brake_force
    return np.array([a[0] / gain, a[1] / cfg.max_yaw_rate])


class ThrottleShield:
    """Closest-point-projection shield behind the throttle interface.

    The shield converts the throttle command to forces with the actuator gains it assumes
    (`shield_config`) and projects in force space; the env then applies the projected throttle with
    its true gains (`env_config`). A mismatch in F_a / F_b is therefore a model error of the shield
    (it mispredicts the force a command produces), not merely a different actuator box.
    """

    def __init__(self, env_config: SeekerUnicycleConfig, shield_config: SeekerUnicycleConfig):
        self.env_config = env_config
        self.shield_config = shield_config
        self.shield = SeekerUnicycleClosestPointProjection(shield_config, SHIELD_GAMMA)
        self.shield_box = gym.spaces.Box(
            low=np.array([-shield_config.max_brake_force, -shield_config.max_yaw_rate]),
            high=np.array([shield_config.max_acc_force, shield_config.max_yaw_rate]),
            dtype=np.float64,
        )

    def _apply(self, force: np.ndarray) -> np.ndarray:
        """Force the shield intends -> force the env actually produces for that command."""
        return to_force(to_throttle(force, self.shield_config), self.env_config)

    def adapt_input(self, observation, info, action, action_space) -> tuple[np.ndarray, bool]:
        force, adapted = self.shield.adapt_input(
            observation, info, to_force(action, self.shield_config), self.shield_box
        )
        return self._apply(force), adapted

    def compute_fail_safe_input(self, observation, info) -> np.ndarray:
        return self._apply(self.shield.compute_fail_safe_input(observation, info))


class ThrottleEnvFactory(InputAdaptionEnvFactory):
    """Exposes THROTTLE_SPACE to the policy; the wrapped env underneath stays in physical units."""

    def create_env(self, mode: EnvMode, seed: int | None = None) -> gym.Env:
        env = super().create_env(mode, seed=seed)
        env.action_space = THROTTLE_SPACE
        return env


def build_env_factory(
    env_config: SeekerUnicycleConfig,
    shield_config: SeekerUnicycleConfig,
    venv_type: VectorEnvType = VectorEnvType.SUBPROC_SHARED_MEM_AUTO,
) -> ThrottleEnvFactory:
    """Registered SeekerUnicycleEnv (true dynamics = `env_config`) wrapped per-env with the
    closest-point-projection shield (assumed dynamics = `shield_config`), commanded by throttle
    (see `ThrottleShield`).

    At F_a = F_b = 2 this is equivalent to commanding forces directly through the symmetric box
    [-2, 2] N, which the base agents were trained with."""
    shield = ThrottleShield(env_config, shield_config)
    factory = ThrottleEnvFactory(
        task=env_config.id,
        venv_type=venv_type,
        **env_config.model_dump(exclude={'id'}),
    )
    factory.set_adaption_fn(shield.adapt_input)
    factory.set_fail_safe_fn(shield.compute_fail_safe_input)
    return factory


def build_ppo_params(cfg: dict) -> PPOParams:
    params = PPOParams()
    for field in PPO_PARAM_FIELDS:
        if cfg.get(field) is not None:
            setattr(params, field, cfg[field])
    return params


def build_ppo_builder(
    cfg: dict,
    env_factory: InputAdaptionEnvFactory,
    *,
    max_epochs: int,
    epoch_num_steps: int,
    logger_factory: LoggerFactory,
    persist_base_dir: str | None = None,
    persistence_enabled: bool = True,  # tianshou only creates a logger when persistence is on
    restore_dir: str | None = None,
    seed: int = 0,
    watch: bool = False,
    num_train_envs: int = 1,
) -> PPOExperimentBuilder:
    """PPO experiment builder mirroring the standard tianshou wiring, with persistence/restore.

    Returns the (unbuilt) builder so callers can still attach a collector factory
    (e.g. `configure_experiment_metrics_logging`) before `.build()`.
    """
    training_config = OnPolicyTrainingConfig(
        max_epochs=max_epochs,
        epoch_num_steps=epoch_num_steps,
        num_training_envs=num_train_envs,
        test_step_num_episodes=10,
        batch_size=cfg.get('batch_size', 64),
        update_step_num_repetitions=cfg.get('update_step_num_repetitions', 1),
    )

    experiment_config = ExperimentConfig(
        seed=seed,
        device='cpu',
        train=True,
        watch=watch,
        persistence_enabled=persistence_enabled,
        persistence_base_dir=persist_base_dir or 'log',
        policy_restore_directory=restore_dir,
        # State-dict (plain tensors) so restore works under torch>=2.6 `weights_only=True`;
        # the full-policy mode can't be unpickled there. Fine-tune restores via this path.
        policy_persistence_mode=PolicyPersistence.Mode.POLICY_STATE_DICT,
    )

    hidden_sizes = [cfg.get('width', 64)] * cfg.get('depth', 2)

    return (
        PPOExperimentBuilder(
            env_factory=env_factory,
            experiment_config=experiment_config,
            training_config=training_config,
        )
        .with_ppo_params(build_ppo_params(cfg))
        .with_logger_factory(logger_factory)
        .with_actor_factory_default(hidden_sizes, torch.nn.ReLU, continuous_unbounded=False)
        .with_critic_factory_default(hidden_sizes, torch.nn.ReLU)
    )
