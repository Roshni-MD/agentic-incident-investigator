from abc import ABC, abstractmethod

from .models import AgentMessage, AgentResponse


class LLMClient(ABC):
    """Provider-independent interface for an LLM."""

    @abstractmethod
    async def generate(
        self,
        messages: list[AgentMessage],
    ) -> AgentResponse:
        """Generate the next agent response."""
        raise NotImplementedError