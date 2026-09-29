from datetime import datetime, timezone
from typing import Any

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

    async def get_metric_history(
        metric_name: str,
        service_name: str,
        start_time: str,
        end_time: str,
        step_seconds: int = 1,
    ) -> list[dict[str, Any]]:
        """Get historical values of a metric for an ML service."""

        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        metrics = repository.get_metrics(
            service_name=service_name,
            start_time=start,
            end_time=end,
        )

        if not hasattr(
            metrics[0] if metrics else None,
            metric_name,
        ):
            return []

        return [
            {
                "timestamp": metric.timestamp.isoformat(),
                "value": getattr(metric, metric_name),
            }
            for metric in metrics
        ]

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

    async def query_logs(
        service_name: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Query logs for an ML service within a time range."""

        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        logs = repository.get_logs(
            service_name=service_name,
            start_time=start,
            end_time=end,
        )

        return [
            {
                "timestamp": log.timestamp.isoformat(),
                "service_name": log.service_name,
                "level": log.level,
                "message": log.message,
                "metadata": log.metadata,
            }
            for log in logs
        ]

    async def get_recent_deployments(
        service_name: str,
        start_time: str,
        end_time: str,
    ) -> list[dict[str, Any]]:
        """Get deployments for an ML service within a time range."""

        start = datetime.fromisoformat(start_time)
        end = datetime.fromisoformat(end_time)

        deployments = repository.get_deployments(
            service_name=service_name,
            start_time=start,
            end_time=end,
        )

        return [
            {
                "deployment_id": deployment.deployment_id,
                "service_name": deployment.service_name,
                "model_name": deployment.model_name,
                "model_version": deployment.model_version,
                "timestamp": deployment.timestamp.isoformat(),
                "previous_version": deployment.previous_version,
            }
            for deployment in deployments
        ]

    registry.register("get_current_metric", get_current_metric)
    registry.register("get_metric_history", get_metric_history)
    registry.register("get_service_health", get_service_health)
    registry.register("query_logs", query_logs)
    registry.register("get_recent_deployments", get_recent_deployments)

    return registry