import json
import os

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
        )

        message = response.choices[0].message

        return AgentResponse(
            answer=message.content or "",
        )