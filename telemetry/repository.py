from datetime import datetime

from .models import Deployment, Incident, LogEvent, MetricSnapshot


class TelemetryRepository:
    """Read-only interface for querying incident telemetry."""

    def __init__(self) -> None:
        self._metrics: dict[str, list[MetricSnapshot]] = {}
        self._logs: dict[str, list[LogEvent]] = {}
        self._deployments: dict[str, list[Deployment]] = {}

    def add_metrics(
        self,
        service_name: str,
        metrics: list[MetricSnapshot],
    ) -> None:
        self._metrics.setdefault(service_name, []).extend(metrics)

    def add_logs(
        self,
        service_name: str,
        logs: list[LogEvent],
    ) -> None:
        self._logs.setdefault(service_name, []).extend(logs)

    def add_deployments(
        self,
        service_name: str,
        deployments: list[Deployment],
    ) -> None:
        self._deployments.setdefault(service_name, []).extend(deployments)

    def get_metrics(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[MetricSnapshot]:
        return [
            metric
            for metric in self._metrics.get(service_name, [])
            if start_time <= metric.timestamp <= end_time
        ]

    def get_logs(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[LogEvent]:
        return [
            log
            for log in self._logs.get(service_name, [])
            if start_time <= log.timestamp <= end_time
        ]

    def get_deployments(
        self,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[Deployment]:
        return [
            deployment
            for deployment in self._deployments.get(service_name, [])
            if start_time <= deployment.timestamp <= end_time
        ]