import inspect
from collections.abc import Awaitable, Callable
from typing import Any, get_type_hints


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

    def schemas(self) -> list[dict[str, Any]]:
        """Return OpenAI-compatible function tool schemas."""

        schemas: list[dict[str, Any]] = []

        for name, function in self._tools.items():
            signature = inspect.signature(function)

            properties: dict[str, Any] = {}
            required: list[str] = []

            type_hints = get_type_hints(function)

            for parameter_name, parameter in signature.parameters.items():
                if parameter_name == "self":
                    continue

                annotation = type_hints.get(
                    parameter_name,
                    parameter.annotation,
                )

                properties[parameter_name] = {
                    "type": self._python_type_to_json_type(annotation),
                }

                if parameter.default is inspect.Parameter.empty:
                    required.append(parameter_name)

            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": inspect.getdoc(function)
                        or f"Execute {name}.",
                        "parameters": {
                            "type": "object",
                            "properties": properties,
                            "required": required,
                            "additionalProperties": False,
                        },
                    },
                }
            )

        return schemas

    @staticmethod
    def _python_type_to_json_type(
        annotation: Any,
    ) -> str:
        """Convert common Python annotations to JSON Schema types."""

        if annotation is str:
            return "string"

        if annotation is int:
            return "integer"

        if annotation is float:
            return "number"

        if annotation is bool:
            return "boolean"

        return "object"