from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class EvidenceSource(str, Enum):
    METRICS = "metrics"
    LOGS = "logs"
    DEPLOYMENT = "deployment"
    CONFIGURATION = "configuration"
    KNOWLEDGE_BASE = "knowledge_base"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


class Evidence(BaseModel):
    source: EvidenceSource
    description: str
    timestamp: datetime | None = None
    strength: EvidenceStrength = EvidenceStrength.MODERATE


class Hypothesis(BaseModel):
    name: str
    confidence: float = Field(ge=0, le=1)
    explanation: str
    evidence: list[Evidence] = []
    recommended_actions: list[str] = Field(default_factory=list)


class TimelineEvent(BaseModel):
    timestamp: datetime
    description: str
    source: EvidenceSource


class InvestigationReport(BaseModel):
    incident_id: str
    service_name: str

    started_at: datetime

    likely_root_cause: str
    confidence: float = Field(ge=0, le=1)

    evidence: list[Evidence] = []
    hypotheses: list[Hypothesis] = []
    timeline: list[TimelineEvent] = []
    recommended_actions: list[str] = []