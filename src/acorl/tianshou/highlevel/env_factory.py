from typing import Any, Self

import gymnasium as gym
from tianshou.highlevel.env import EnvFactoryRegistered, EnvMode

from acorl.common.definitions import FailSafeInputFn, InputAdaptionFnEnv
from acorl.common.env_wrapper.adaption import InputAdaptionEnvWrapper


class InputAdaptionEnvFactory(EnvFactoryRegistered):
    def set_env_wrapper_kwargs(self, kwargs: dict[str, Any]) -> Self:
        self.env_wrapper_kwargs = kwargs
        return self

    def set_fail_safe_fn(self, fn: FailSafeInputFn) -> Self:
        self.fail_safe_fn = fn
        return self

    def set_adaption_fn(self, fn: InputAdaptionFnEnv) -> Self:
        self.adaption_fn = fn
        return self

    def create_env(self, mode: EnvMode, seed: int | None = None) -> gym.Env:
        assert self.fail_safe_fn is not None, 'Fail safe function is not set.'
        assert self.adaption_fn is not None, 'Adaption function is not set.'

        if not hasattr(self, 'env_wrapper_kwargs'):
            self.env_wrapper_kwargs = {}

        env = super().create_env(mode, seed=seed)
        return InputAdaptionEnvWrapper(
            env, self.adaption_fn, self.fail_safe_fn, **self.env_wrapper_kwargs
        )
