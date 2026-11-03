from typing import Any, Protocol

import pygame


class PygameRenderHookFn[ObsType](Protocol):  # pyright: ignore[reportInvalidTypeVarUse]
    def __call__(self, observation: ObsType, info: dict[str, Any], surface: pygame.Surface): ...
