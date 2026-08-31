from datetime import datetime, timezone

from telemetry.repository import TelemetryRepository

from .tools import AgentToolRegistry


def build_telemetry_tool_registry(
    repository: TelemetryRepository,
) -> AgentToolRegistry:
    """Build the tools available to the investigation agent."""

    registry = AgentToolRegistry()

    async def get_current_metric(
        metric_name: str,
        service_name: str,
    ) -> dict[str, object]:
        """Get the latest value of a metric for an ML service."""

        metrics = repository.get_metrics(
            service_name=service_name,
            start_time=datetime.min.replace(tzinfo=timezone.utc),
            end_time=datetime.max.replace(tzinfo=timezone.utc),
        )

        if not metrics:
            return {
                "service_name": service_name,
                "metric_name": metric_name,
                "status": "not_found",
            }

        latest = metrics[-1]

        if not hasattr(latest, metric_name):
            return {
                "service_name": service_name,
                "metric_name": metric_name,
                "status": "unknown_metric",
            }

        return {
            "service_name": service_name,
            "metric_name": metric_name,
            "value": getattr(latest, metric_name),
            "timestamp": latest.timestamp.isoformat(),
            "status": "ok",
        }

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        """Get the latest telemetry metrics for an ML service."""

        metrics = repository.get_metrics(
            service_name=service_name,
            start_time=datetime.min.replace(tzinfo=timezone.utc),
            end_time=datetime.max.replace(tzinfo=timezone.utc),
        )

        if not metrics:
            return {
                "service_name": service_name,
                "status": "not_found",
            }

        latest = metrics[-1]

        return {
            "service_name": service_name,
            "status": "ok",
            "metrics": {
                "cpu_utilization": {
                    "value": latest.cpu_utilization,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "gpu_utilization": {
                    "value": latest.gpu_utilization,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "gpu_memory_utilization": {
                    "value": latest.gpu_memory_utilization,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "p95_latency_ms": {
                    "value": latest.p95_latency_ms,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "throughput_rps": {
                    "value": latest.throughput_rps,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "data_loading_ms": {
                    "value": latest.data_loading_ms,
                    "timestamp": latest.timestamp.isoformat(),
                },
                "gpu_kernel_ms": {
                    "value": latest.gpu_kernel_ms,
                    "timestamp": latest.timestamp.isoformat(),
                },
            },
        }

    registry.register(
        "get_current_metric",
        get_current_metric,
    )

    registry.register(
        "get_service_health",
        get_service_health,
    )

    return registry