from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class IncidentType(str, Enum):
    CPU_BOTTLENECK = "cpu_bottleneck"
    GPU_OOM = "gpu_oom"
    NETWORK_BOTTLENECK = "network_bottleneck"
    MODEL_LATENCY = "model_latency"


class MetricSnapshot(BaseModel):
    timestamp: datetime

    cpu_utilization: float = Field(ge=0, le=100)
    gpu_utilization: float = Field(ge=0, le=100)

    gpu_memory_utilization: float = Field(ge=0, le=100)

    p95_latency_ms: float = Field(ge=0)
    throughput_rps: float = Field(ge=0)

    data_loading_ms: float = Field(ge=0)
    gpu_kernel_ms: float = Field(ge=0)
    network_utilization: float = Field(ge=0, le=100)
    network_latency_ms: float = Field(ge=0)


class Deployment(BaseModel):
    deployment_id: str
    service_name: str
    model_name: str
    model_version: str

    timestamp: datetime

    previous_version: str | None = None


class LogEvent(BaseModel):
    timestamp: datetime
    service_name: str
    level: str
    message: str

    metadata: dict[str, str] = {}


class Incident(BaseModel):
    incident_id: str
    incident_type: IncidentType

    service_name: str

    started_at: datetime

    ground_truth: str

    metrics: list[MetricSnapshot] = []
    logs: list[LogEvent] = []
    deployments: list[Deployment] = []