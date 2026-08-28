import time

from prometheus_client import start_http_server

from .prometheus_metrics import publish_metrics
from .simulator import create_cpu_bottleneck_incident


def publish_snapshot(service_name, snapshot) -> None:
    publish_metrics(
        service_name=service_name,
        cpu_utilization=snapshot.cpu_utilization,
        gpu_utilization=snapshot.gpu_utilization,
        gpu_memory_utilization=snapshot.gpu_memory_utilization,
        p95_latency_ms=snapshot.p95_latency_ms,
        throughput_rps=snapshot.throughput_rps,
        data_loading_ms=snapshot.data_loading_ms,
        gpu_kernel_ms=snapshot.gpu_kernel_ms,
    )


def replay_incident(
    interval_seconds: float = 2.0,
) -> None:
    incident = create_cpu_bottleneck_incident()

    print(
        f"Replaying incident {incident.incident_id} "
        f"for {incident.service_name}"
    )

    for index, snapshot in enumerate(incident.metrics, start=1):
        publish_snapshot(
            incident.service_name,
            snapshot,
        )

        print(
            f"[{index}/{len(incident.metrics)}] "
            f"CPU={snapshot.cpu_utilization}% "
            f"GPU={snapshot.gpu_utilization}% "
            f"Latency={snapshot.p95_latency_ms}ms "
            f"Throughput={snapshot.throughput_rps} rps"
        )

        time.sleep(interval_seconds)

    print("Incident replay complete.")


if __name__ == "__main__":
    start_http_server(8000)

    print("Prometheus metrics available at http://localhost:8000/metrics")

    replay_incident()

    input("Press Enter to stop...\n")