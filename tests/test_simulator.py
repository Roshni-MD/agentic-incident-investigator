from telemetry.simulator import create_cpu_bottleneck_incident
from telemetry.models import IncidentType


def test_cpu_bottleneck_incident():

    incident = create_cpu_bottleneck_incident()

    assert incident.incident_id == "INC-1001"

    assert incident.incident_type == IncidentType.CPU_BOTTLENECK

    assert incident.ground_truth == "CPU-side preprocessing bottleneck"

    assert len(incident.metrics) == 10

    assert len(incident.logs) == 2

    assert len(incident.deployments) == 1


def test_cpu_bottleneck_has_expected_signal():

    incident = create_cpu_bottleneck_incident()

    normal = incident.metrics[0]
    degraded = incident.metrics[-1]

    assert normal.gpu_utilization > degraded.gpu_utilization

    assert degraded.cpu_utilization > normal.cpu_utilization

    assert degraded.p95_latency_ms > normal.p95_latency_ms

    assert degraded.throughput_rps < normal.throughput_rps

    assert degraded.data_loading_ms > normal.data_loading_ms