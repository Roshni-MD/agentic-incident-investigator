from telemetry.exporter import publish_snapshot
from telemetry.simulator import create_cpu_bottleneck_incident


def test_publish_snapshot():
    incident = create_cpu_bottleneck_incident()
    snapshot = incident.metrics[-1]

    # The test verifies the adapter accepts a MetricSnapshot
    # and publishes it without raising an exception.
    publish_snapshot(
        service_name=incident.service_name,
        snapshot=snapshot,
    )