from telemetry.models import Incident

from .llm import LLMClient
from .models import (
    AgentMessage,
    AgentResponse,
)
from .tools import AgentToolRegistry


class AgentRunner:
    """Runs the investigation agent tool-calling loop."""

    def __init__(
        self,
        llm: LLMClient,
        tools: AgentToolRegistry,
        max_iterations: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations

    async def run(
        self,
        incident: Incident,
    ) -> AgentResponse:
        messages = [
            AgentMessage(
                role="system",
                content=(
                    "You are an ML infrastructure incident "
                    "investigation agent. Investigate incidents "
                    "using the available telemetry tools. "
                    "Gather evidence before determining the "
                    "root cause."
                ),
            ),
            AgentMessage(
                role="user",
                content=(
                    f"Investigate incident {incident.incident_id} "
                    f"for service {incident.service_name}."
                ),
            ),
        ]

        for _ in range(self.max_iterations):
            response = await self.llm.generate(
                messages,
                tools=self.tools.schemas(),
            )

            if not response.tool_calls:
                return response

            for tool_call in response.tool_calls:
                tool = self.tools.get(tool_call.tool_name)

                result = await tool(
                    **tool_call.arguments,
                )

                messages.append(
                    AgentMessage(
                        role="assistant",
                        content=(
                            f"Tool call: {tool_call.tool_name}"
                        ),
                    )
                )

                messages.append(
                    AgentMessage(
                        role="tool",
                        content=str(result),
                    )
                )

        raise RuntimeError(
            "Agent exceeded maximum number of iterations."
        )