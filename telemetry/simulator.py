from datetime import datetime, timedelta, timezone

from .models import (
    Deployment,
    Incident,
    IncidentType,
    LogEvent,
    MetricSnapshot,
)


def create_cpu_bottleneck_incident(
    start: datetime | None = None,
) -> Incident:
    if start is None:
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
                network_utilization=40.0,
                network_latency_ms=5.0,
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
                network_utilization=40.0,
                network_latency_ms=5.0,
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

def create_gpu_oom_incident(
    start: datetime | None = None,
) -> Incident:
    if start is None:
        start = datetime.now(timezone.utc)

    metrics = [
        MetricSnapshot(
            timestamp=start - timedelta(minutes=2),
            cpu_utilization=50,
            gpu_utilization=95,
            gpu_memory_utilization=80,
            p95_latency_ms=80,
            throughput_rps=1200,
            data_loading_ms=20,
            gpu_kernel_ms=90,
            network_utilization=40.0,
            network_latency_ms=5.0,
        ),
        MetricSnapshot(
            timestamp=start - timedelta(minutes=1),
            cpu_utilization=55,
            gpu_utilization=20,
            gpu_memory_utilization=99,
            p95_latency_ms=500,
            throughput_rps=300,
            data_loading_ms=25,
            gpu_kernel_ms=120,
            network_utilization=40.0,
            network_latency_ms=5.0,
        ),
    ]

    logs = [
        LogEvent(
            timestamp=start - timedelta(minutes=1),
            service_name="image-ranking-service",
            level="ERROR",
            message="CUDA out of memory",
        ),
    ]

    deployments = [
        Deployment(
            deployment_id="deploy-gpu-oom",
            service_name="image-ranking-service",
            model_name="ranking-model",
            model_version="v3.0",
            previous_version="v2.9",
            timestamp=start - timedelta(minutes=2),
        ),
    ]

    return Incident(
        incident_id="INC-GPU-OOM",
        incident_type=IncidentType.GPU_OOM,
        service_name="image-ranking-service",
        started_at=metrics[0].timestamp,
        ground_truth="GPU memory exhaustion",
        metrics=metrics,
        logs=logs,
        deployments=deployments,
    )

def create_network_bottleneck_incident(
    start: datetime | None = None,
) -> Incident:
    if start is None:
        start = datetime.now(timezone.utc)

    metrics = [
        MetricSnapshot(
            timestamp=start - timedelta(minutes=5),
            cpu_utilization=55.0,
            gpu_utilization=90.0,
            gpu_memory_utilization=68.0,
            p95_latency_ms=80.0,
            throughput_rps=1200.0,
            data_loading_ms=20.0,
            gpu_kernel_ms=90.0,
            network_utilization=35.0,
            network_latency_ms=5.0,
        )
        for _ in range(5)
    ]

    degraded_metrics = [
        MetricSnapshot(
            timestamp=start,
            cpu_utilization=55.0,
            gpu_utilization=45.0,
            gpu_memory_utilization=70.0,
            p95_latency_ms=180.0,
            throughput_rps=650.0,
            data_loading_ms=100.0,
            gpu_kernel_ms=90.0,
            network_utilization=97.0,
            network_latency_ms=250.0,
        )
        for _ in range(5)
    ]

    metrics.extend(degraded_metrics)

    logs = [
        LogEvent(
            timestamp=start,
            service_name="image-ranking-service",
            level="WARNING",
            message="Network latency increased significantly",
            metadata={
                "network_latency_ms": "250",
            },
        ),
        LogEvent(
            timestamp=start + timedelta(minutes=1),
            service_name="image-ranking-service",
            level="WARNING",
            message="Network utilization approaching saturation",
            metadata={
                "network_utilization": "97",
            },
        ),
    ]

    deployments = [
        Deployment(
            deployment_id="deploy-network-1001",
            service_name="image-ranking-service",
            model_name="ranking-model",
            model_version="v3.1",
            timestamp=start - timedelta(minutes=1),
            previous_version="v3.0",
        )
    ]

    return Incident(
        incident_id="INC-NET-1001",
        incident_type=IncidentType.NETWORK_BOTTLENECK,
        service_name="image-ranking-service",
        started_at=metrics[5].timestamp,
        ground_truth="Network bottleneck",
        metrics=metrics,
        logs=logs,
        deployments=deployments,
    )


if __name__ == "__main__":
    incident = create_cpu_bottleneck_incident()

    print(incident.model_dump_json(indent=2))