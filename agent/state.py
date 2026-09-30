from pydantic import BaseModel, Field

from .models import AgentMessage, AgentResponse


class AgentState(BaseModel):
    """Mutable state for a single agent investigation."""

    messages: list[AgentMessage] = Field(default_factory=list)
    iteration: int = 0

    def add_message(self, message: AgentMessage) -> None:
        """Add a message to the investigation history."""
        self.messages.append(message)

    def add_assistant_response(self, response: AgentResponse) -> None:
        """Add an LLM response to the investigation history."""
        self.add_message(
            AgentMessage(
                role="assistant",
                content=response.answer,
                tool_calls=response.tool_calls,
            )
        )

    def add_tool_result(
        self,
        content: str,
        tool_call_id: str | None = None,
    ) -> None:
        """Add a tool result to the investigation history."""
        self.add_message(
            AgentMessage(
                role="tool",
                content=content,
                tool_call_id=tool_call_id,
            )
        )

    def next_iteration(self) -> None:
        """Advance the investigation iteration."""
        self.iteration += 1