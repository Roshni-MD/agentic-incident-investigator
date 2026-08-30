from .models import Incident
from .repository import TelemetryRepository
from .simulator import (
    create_cpu_bottleneck_incident,
    create_gpu_oom_incident,
    create_network_bottleneck_incident,
)


def load_cpu_bottleneck_scenario() -> tuple[Incident, TelemetryRepository]:
    incident = create_cpu_bottleneck_incident()

    repository = TelemetryRepository()

    repository.add_metrics(
        incident.service_name,
        incident.metrics,
    )

    repository.add_logs(
        incident.service_name,
        incident.logs,
    )

    repository.add_deployments(
        incident.service_name,
        incident.deployments,
    )

    return incident, repository

def load_gpu_oom_scenario() -> tuple[Incident, TelemetryRepository]:
    incident = create_gpu_oom_incident()

    repository = TelemetryRepository()

    repository.add_metrics(
        incident.service_name,
        incident.metrics,
    )

    repository.add_logs(
        incident.service_name,
        incident.logs,
    )

    repository.add_deployments(
        incident.service_name,
        incident.deployments,
    )

    return incident, repository

def load_network_bottleneck_scenario() -> tuple[Incident, TelemetryRepository]:
    incident = create_network_bottleneck_incident()

    repository = TelemetryRepository()

    repository.add_metrics(
        incident.service_name,
        incident.metrics,
    )

    repository.add_logs(
        incident.service_name,
        incident.logs,
    )

    repository.add_deployments(
        incident.service_name,
        incident.deployments,
    )

    return incident, repository