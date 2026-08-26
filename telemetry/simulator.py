from datetime import datetime, timedelta, timezone

from .models import (
    Deployment,
    Incident,
    IncidentType,
    LogEvent,
    MetricSnapshot,
)


def create_cpu_bottleneck_incident() -> Incident:
    start = datetime.now(timezone.utc)

    metrics = []

    # Normal operating period
    for i in range(5):
        metrics.append(
            MetricSnapshot(
                timestamp=start + timedelta(minutes=i),
                cpu_utilization=55,
                gpu_utilization=91,
                gpu_memory_utilization=68,
                p95_latency_ms=80,
                throughput_rps=1200,
                data_loading_ms=20,
                gpu_kernel_ms=90,
            )
        )

    # Incident begins
    for i in range(5, 10):
        metrics.append(
            MetricSnapshot(
                timestamp=start + timedelta(minutes=i),
                cpu_utilization=96,
                gpu_utilization=42,
                gpu_memory_utilization=70,
                p95_latency_ms=135,
                throughput_rps=760,
                data_loading_ms=380,
                gpu_kernel_ms=90,
            )
        )

    deployments = [
        Deployment(
            deployment_id="deploy-1002",
            service_name="image-ranking-service",
            model_name="ranking-model",
            model_version="v2.4",
            previous_version="v2.3",
            timestamp=start + timedelta(minutes=4),
        )
    ]

    logs = [
        LogEvent(
            timestamp=start + timedelta(minutes=5),
            service_name="image-ranking-service",
            level="WARNING",
            message="Data preprocessing latency increased",
            metadata={
                "component": "image-preprocessor",
                "version": "v2.4",
            },
        ),
        LogEvent(
            timestamp=start + timedelta(minutes=6),
            service_name="image-ranking-service",
            level="WARNING",
            message="GPU utilization below expected threshold",
            metadata={
                "gpu_utilization": "42",
            },
        ),
    ]

    return Incident(
        incident_id="INC-1001",
        incident_type=IncidentType.CPU_BOTTLENECK,
        service_name="image-ranking-service",
        started_at=start + timedelta(minutes=5),
        ground_truth="CPU-side preprocessing bottleneck",
        metrics=metrics,
        logs=logs,
        deployments=deployments,
    )


if __name__ == "__main__":
    incident = create_cpu_bottleneck_incident()

    print(incident.model_dump_json(indent=2))