from typing import Any

from .models import AgentToolCall
from .tools import AgentToolRegistry


class ToolExecutor:
    """Executes agent tool calls using the registered tools."""

    def __init__(self, registry: AgentToolRegistry) -> None:
        self.registry = registry

    async def execute(self, tool_call: AgentToolCall) -> Any:
        tool = self.registry.get(tool_call.tool_name)
        return await tool(**tool_call.arguments)