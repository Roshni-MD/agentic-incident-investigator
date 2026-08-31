from pydantic import BaseModel, Field


class AgentMessage(BaseModel):
    role: str
    content: str


class AgentToolCall(BaseModel):
    tool_name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class AgentToolResult(BaseModel):
    tool_name: str
    result: object


class AgentResponse(BaseModel):
    answer: str = ""
    tool_calls: list[AgentToolCall] = Field(default_factory=list)


class InvestigationContext(BaseModel):
    incident_id: str
    service_name: str
    incident_type: str
    started_at: str

    available_tools: list[str] = Field(
        default_factory=list,
    )


