from abc import ABC, abstractmethod
from typing import Any

from .models import AgentMessage, AgentResponse


class LLMClient(ABC):
    """Provider-independent interface for an LLM."""

    @abstractmethod
    async def generate(
        self,
        messages: list[AgentMessage],
        tools: list[dict[str, Any]] | None = None,
    ) -> AgentResponse:
        """Generate the next agent response."""
        raise NotImplementedError