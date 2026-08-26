from prometheus_client import start_http_server

from .prometheus_metrics import publish_metrics
from .simulator import create_cpu_bottleneck_incident


def export_incident_metrics() -> None:

    incident = create_cpu_bottleneck_incident()

    # Publish the final telemetry snapshot.
    latest = incident.metrics[-1]

    publish_metrics(
        service_name=incident.service_name,
        cpu_utilization=latest.cpu_utilization,
        gpu_utilization=latest.gpu_utilization,
        gpu_memory_utilization=latest.gpu_memory_utilization,
        p95_latency_ms=latest.p95_latency_ms,
        throughput_rps=latest.throughput_rps,
        data_loading_ms=latest.data_loading_ms,
        gpu_kernel_ms=latest.gpu_kernel_ms,
    )


if __name__ == "__main__":

    start_http_server(8000)

    export_incident_metrics()

    print("Prometheus metrics available at http://localhost:8000/metrics")

    input("Press Enter to stop...\n")