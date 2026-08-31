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