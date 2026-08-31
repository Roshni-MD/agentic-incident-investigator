from collections.abc import Awaitable, Callable
from typing import Any


ToolFunction = Callable[..., Awaitable[Any]]


class AgentToolRegistry:
    """Registry of tools available to the investigation agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(
        self,
        name: str,
        function: ToolFunction,
    ) -> None:
        self._tools[name] = function

    def get(self, name: str) -> ToolFunction:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())