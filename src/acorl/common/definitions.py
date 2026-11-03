import dataclasses
from abc import ABC, abstractmethod
from typing import Any, Protocol

import gymnasium as gym
import numpy as np


@dataclasses.dataclass
class BaseEnvConfig:
    id: str
    render_mode: str | None = None

    def model_dump(self, exclude: set | None = None, include: set | None = None) -> dict:
        assert exclude is None or include is None
        d = dataclasses.asdict(self)
        if include:
            return {k: v for k, v in d.items() if k in include}
        if exclude:
            return {k: v for k, v in d.items() if k not in exclude}
        return d


class FailSafeInputFn[ObsType, ActType](Protocol):  # pyright: ignore[reportInvalidTypeVarUse]
    def __call__(self, observation: ObsType, info: dict[str, Any]) -> ActType: ...


class InputAdaptionFnEnv(Protocol):
    """
    Input adaption fn interface in the env wrapper.
    """

    def __call__(
        self,
        observation: np.ndarray,
        info: dict[str, Any],
        action: np.ndarray,
        action_space: gym.spaces.Box,
    ) -> tuple[np.ndarray, bool]: ...


class InputAdaptionCalculator[ObsType, ActType](ABC):
    @abstractmethod
    def adapt_input(
        self,
        observation: ObsType,
        info: dict[str, Any],
        action: ActType,
        action_space: gym.spaces.Box,
    ) -> tuple[ActType, bool]:
        pass

    @abstractmethod
    def compute_fail_safe_input(self, observation: ObsType, info: dict[str, Any]) -> ActType:
        pass
