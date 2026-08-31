from collections.abc import Awaitable, Callable
from typing import Any


ToolFunction = Callable[..., Awaitable[Any]]


class AgentToolRegistry:
    """Registry of tools available to the investigation agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}
        self._schemas: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        function: ToolFunction,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._tools[name] = function

        if schema is not None:
            self._schemas[name] = schema

    def get(self, name: str) -> ToolFunction:
        return self._tools[name]

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def schemas(self) -> list[dict[str, Any]]:
        return list(self._schemas.values())