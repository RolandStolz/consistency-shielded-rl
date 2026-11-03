from typing import Any, SupportsFloat

import gymnasium as gym
from gymnasium.core import ActType, Env, ObsType

from acorl.common.definitions import FailSafeInputFn, InputAdaptionFnEnv
from acorl.common.exceptions import InputAdaptionComputationError


class InputAdaptionEnvWrapper(gym.Wrapper):
    """
    For implementations that don't explicitly compute the relevant input set.
    """

    def __init__(
        self,
        env: Env[ObsType, ActType],
        input_adaption_fn: InputAdaptionFnEnv,
        fail_safe_fn: FailSafeInputFn,
    ):
        super().__init__(env)
        assert isinstance(self.env.action_space, gym.spaces.Box)
        self._input_adaption_fn = input_adaption_fn
        self._fail_safe_fn = fail_safe_fn
        self._previous_obs: ObsType = None  # type: ignore
        self._previous_info: dict[str, Any] = None  # type: ignore

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._previous_obs = obs
        self._previous_info = info
        return obs, info

    def step(self, action: Any) -> tuple[Any, SupportsFloat, bool, bool, dict[str, Any]]:
        try:
            adapted_action, required_intervention = self._input_adaption_fn(
                self._previous_obs,
                self._previous_info,
                action,
                self.env.action_space,  # pyright: ignore[reportArgumentType]
            )
        except InputAdaptionComputationError:
            gym.logger.warn('Error in the input adaption function.')
            adapted_action = self._fail_safe_fn(self._previous_obs, self._previous_info)
            required_intervention = True

        obs, reward, terminated, truncated, info = self.env.step(adapted_action)

        self._previous_obs = obs
        self._previous_info = info
        info['adapted_action'] = adapted_action
        info['required_intervention'] = required_intervention

        return obs, reward, terminated, truncated, info
