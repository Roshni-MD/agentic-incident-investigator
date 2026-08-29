from datetime import datetime, timezone
from typing import Any

import re
import requests
from pydantic import BaseModel


SERVICE_NAME_PATTERN = re.compile(
    r"^[a-zA-Z0-9_-]+$"
)


class MetricPoint(BaseModel):
    timestamp: datetime
    value: float


class PrometheusClient:
    """Small client for querying the Prometheus HTTP API."""

    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")

    def query(self, promql: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/v1/query",
            params={"query": promql},
            timeout=5,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "success":
            raise RuntimeError(
                f"Prometheus query failed: {payload}"
            )

        return payload["data"]

    def query_range(
        self,
        promql: str,
        start_time: datetime,
        end_time: datetime,
        step_seconds: int = 1,
    ) -> dict[str, Any]:

        response = requests.get(
            f"{self.base_url}/api/v1/query_range",
            params={
                "query": promql,
                "start": start_time.timestamp(),
                "end": end_time.timestamp(),
                "step": step_seconds,
            },
            timeout=5,
        )

        response.raise_for_status()

        payload = response.json()

        if payload.get("status") != "success":
            raise RuntimeError(
                f"Prometheus range query failed: {payload}"
            )

        return payload["data"]


METRIC_MAPPING = {
    "cpu_utilization": "ml_service_cpu_utilization",
    "gpu_utilization": "ml_service_gpu_utilization",
    "gpu_memory_utilization": "ml_service_gpu_memory_utilization",
    "p95_latency_ms": "ml_service_p95_latency_ms",
    "throughput_rps": "ml_service_throughput_rps",
    "data_loading_ms": "ml_service_data_loading_ms",
    "gpu_kernel_ms": "ml_service_gpu_kernel_ms",
}


def get_metric_name(metric_name: str) -> str:
    try:
        return METRIC_MAPPING[metric_name]
    except KeyError:
        raise ValueError(
            f"Unsupported metric: {metric_name}. "
            f"Supported metrics: {list(METRIC_MAPPING)}"
        )


def build_metric_query(
    metric_name: str,
    service_name: str,
) -> str:
    
    if not SERVICE_NAME_PATTERN.fullmatch(service_name):
        raise ValueError(
            f"Invalid service name: {service_name}"
        )

    prometheus_metric = get_metric_name(metric_name)

    return (
        f'{prometheus_metric}'
        f'{{service="{service_name}"}}'
    )

def get_metric_history(
    client: PrometheusClient,
    metric_name: str,
    service_name: str,
    start_time: datetime,
    end_time: datetime,
    step_seconds: int = 1,
) -> list[MetricPoint]:

    promql = build_metric_query(
        metric_name=metric_name,
        service_name=service_name,
    )

    data = client.query_range(
        promql=promql,
        start_time=start_time,
        end_time=end_time,
        step_seconds=step_seconds,
    )

    return normalize_query_result(data)

def normalize_query_result(
    data: dict[str, Any],
) -> list[MetricPoint]:

    points = []

    for result in data.get("result", []):

        values = result.get("values")

        if values is None:
            value = result.get("value")

            if value is not None:
                values = [value]

        if not values:
            continue

        for timestamp, value in values:
            points.append(
                MetricPoint(
                    timestamp=datetime.fromtimestamp(
                        float(timestamp),
                        tz=timezone.utc,
                    ),
                    value=float(value),
                )
            )

    return points