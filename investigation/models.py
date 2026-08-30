from datetime import datetime

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source: str
    description: str
    timestamp: datetime | None = None


class Hypothesis(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[Evidence] = []
    recommended_actions: list[str] = Field(default_factory=list)


class InvestigationReport(BaseModel):
    incident_id: str
    service_name: str

    started_at: datetime

    likely_root_cause: str
    confidence: float = Field(ge=0, le=1)

    evidence: list[Evidence] = []
    hypotheses: list[Hypothesis] = []
    recommended_actions: list[str] = []