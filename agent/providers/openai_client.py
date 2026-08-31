import json
import os

from typing import Any

from openai import AsyncOpenAI

from agent.llm import LLMClient
from agent.models import (
    AgentMessage,
    AgentResponse,
    AgentToolCall,
)


class OpenAIClient(LLMClient):
    """OpenAI-backed implementation of the LLMClient interface."""

    def __init__(
        self,
        model: str = "gpt-5-mini",
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set."
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
        )

        self.model = model

    async def generate(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in messages
            ],
            tools=tools or None,
        )

        message = response.choices[0].message

        tool_calls: list[AgentToolCall] = []

        for tool_call in getattr(message, "tool_calls", None) or []:
            arguments = json.loads(
                tool_call.function.arguments or "{}"
            )

            tool_calls.append(
                AgentToolCall(
                    tool_name=tool_call.function.name,
                    arguments=arguments,
                    tool_call_id=tool_call.id,
                )
            )

        return AgentResponse(
            answer=message.content or "",
            tool_calls=tool_calls,
        )