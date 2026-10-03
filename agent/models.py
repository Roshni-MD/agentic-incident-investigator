from pydantic import BaseModel, Field


class AgentToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, object] = Field(
        default_factory=dict,
    )
    tool_call_id: str | None = None


class AgentToolResult(BaseModel):
    tool_name: str
    result: object
    tool_call_id: str | None = None


class AgentMessage(BaseModel):
    role: str
    content: str = ""

    tool_call_id: str | None = None
    tool_calls: list[AgentToolCall] = Field(
        default_factory=list,
    )


class AgentResponse(BaseModel):
    answer: str = ""
    tool_calls: list[AgentToolCall] = Field(
        default_factory=list,
    )


class InvestigationContext(BaseModel):
    incident_id: str
    service_name: str
    incident_type: str
    started_at: str

    available_tools: list[str] = Field(
        default_factory=list,
    )


class InvestigationEvidence(BaseModel):
    """Evidence collected during an investigation."""

    source: str
    description: str
    value: object | None = None


class InvestigationFinding(BaseModel):
    hypothesis: str
    confidence: float = Field(ge=0, le=1)
    explanation: str = ""
    evidence: list[InvestigationEvidence] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)