import asyncio
from datetime import datetime
from typing import Any

from mcp.server.mcpserver import MCPServer

from telemetry.query import (
    PrometheusClient,
    get_current_metric as query_current_metric,
    get_metric_history,
)


mcp = MCPServer(
    name="AI Incident Investigation Agent",
    version="0.1.0",
    description=(
        "MCP server providing observability tools for "
        "ML service incident investigation."
    ),
)

prometheus = PrometheusClient()


async def get_current_metric(
    metric_name: str,
    service_name: str,
) -> dict[str, Any]:
    """Get the current value of a supported metric."""

    point = query_current_metric(
        client=prometheus,
        metric_name=metric_name,
        service_name=service_name,
    )

    if point is None:
        return {
            "service_name": service_name,
            "metric_name": metric_name,
            "value": None,
            "timestamp": None,
            "status": "no_data",
        }

    return {
        "service_name": service_name,
        "metric_name": metric_name,
        "value": point.value,
        "timestamp": point.timestamp.isoformat(),
        "status": "ok",
    }


async def get_metric_history_tool(
    metric_name: str,
    service_name: str,
    start_time: str,
    end_time: str,
    step_seconds: int = 1,
) -> list[dict[str, Any]]:
    """Get historical values for a supported metric."""

    start = datetime.fromisoformat(start_time)
    end = datetime.fromisoformat(end_time)

    points = get_metric_history(
        client=prometheus,
        metric_name=metric_name,
        service_name=service_name,
        start_time=start,
        end_time=end,
        step_seconds=step_seconds,
    )

    return [
        {
            "timestamp": point.timestamp.isoformat(),
            "value": point.value,
        }
        for point in points
    ]


async def get_service_health(
    service_name: str,
) -> dict[str, Any]:
    """Get the current observability snapshot for an ML service."""

    metric_names = [
        "cpu_utilization",
        "gpu_utilization",
        "gpu_memory_utilization",
        "p95_latency_ms",
        "throughput_rps",
        "data_loading_ms",
        "gpu_kernel_ms",
    ]

    metrics = {}

    for metric_name in metric_names:
        point = query_current_metric(
            client=prometheus,
            metric_name=metric_name,
            service_name=service_name,
        )

        if point is None:
            metrics[metric_name] = {
                "value": None,
                "timestamp": None,
            }
        else:
            metrics[metric_name] = {
                "value": point.value,
                "timestamp": point.timestamp.isoformat(),
            }

    return {
        "service_name": service_name,
        "status": "ok",
        "metrics": metrics,
    }


mcp.add_tool(
    get_current_metric,
    name="get_current_metric",
    description="Get the current value of a metric for an ML service.",
)

mcp.add_tool(
    get_metric_history_tool,
    name="get_metric_history",
    description="Get historical values of a metric for an ML service.",
)

mcp.add_tool(
    get_service_health,
    name="get_service_health",
    description="Get the current observability snapshot for an ML service.",
)


async def main() -> None:
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())