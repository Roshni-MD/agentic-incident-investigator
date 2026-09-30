from telemetry.models import Incident

from .context import build_investigation_context, build_system_prompt
from .llm import LLMClient
from .models import (
    AgentMessage,
    AgentResponse
)
from .tools import AgentToolRegistry
from .tool_executor import ToolExecutor
from .state import AgentState

import json


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
        self.tool_executor = ToolExecutor(tools)
        self.max_iterations = max_iterations

    async def run(
        self,
        incident: Incident,
    ) -> AgentResponse:

        context = build_investigation_context(
            incident=incident,
            tools=self.tools,
        )

        state = AgentState(
            messages = [
                AgentMessage(
                    role="system",
                    content=build_system_prompt(context),
                ),
                AgentMessage(
                    role="user",
                    content=(
                        f"Investigate incident {incident.incident_id} "
                        f"for service {incident.service_name}."
                    ),
                ),
            ]
        )

        for _ in range(self.max_iterations):
            state.next_iteration()

            response = await self.llm.generate(
                state.messages,
                tools=self.tools.schemas(),
            )

            if not response.tool_calls:
                return response

            state.add_assistant_response(response)

            for tool_call in response.tool_calls:
                result = await self.tool_executor.execute(tool_call)

                state.add_tool_result(
                    content=json.dumps(result),
                    tool_call_id=tool_call.tool_call_id,
                )

        raise RuntimeError(
            "Agent exceeded maximum number of iterations."
        )