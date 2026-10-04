from pydantic import BaseModel, Field
from enum import Enum

from .models import AgentMessage, AgentResponse, InvestigationFinding, InvestigationEvidence


class InvestigationStatus(str, Enum):
    """Lifecycle state of an investigation."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentState(BaseModel):
    """Mutable state for a single agent investigation."""

    messages: list[AgentMessage] = Field(default_factory=list)
    iteration: int = 0
    status: InvestigationStatus = InvestigationStatus.RUNNING
    evidence: list[InvestigationEvidence] = Field(default_factory=list)

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

    def mark_completed(self) -> None:
        """Mark the investigation as successfully completed."""
        self.status = InvestigationStatus.COMPLETED

    def mark_failed(self) -> None:
        """Mark the investigation as failed."""
        self.status = InvestigationStatus.FAILED

    def add_evidence(self, evidence: InvestigationEvidence) -> None:
        """Add evidence collected during the investigation."""
        self.evidence.append(evidence)
    
class AgentRunResult(BaseModel):
    """Result of an agent investigation."""

    answer: str
    state: AgentState
    findings: list[InvestigationFinding] = Field(default_factory=list)
    formatted_report: str = ""