import asyncio
from datetime import datetime
from typing import Any

from mcp.server.mcpserver import MCPServer

from telemetry.query import (
    PrometheusClient,
    get_current_metric as query_current_metric,
    get_metric_history,
)

from telemetry.scenarios import load_cpu_bottleneck_scenario


mcp = MCPServer(
    name="AI Incident Investigation Agent",
    version="0.1.0",
    description=(
        "MCP server providing observability tools for "
        "ML service incident investigation."
    ),
)

prometheus = PrometheusClient()

incident, repository = load_cpu_bottleneck_scenario()

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

mcp.add_tool(
    query_logs,
    name="query_logs",
    description="Query logs for an ML service within a time range.",
)

mcp.add_tool(
    get_recent_deployments,
    name="get_recent_deployments",
    description=(
        "Get deployments for an ML service within a time range."
    ),
)


async def main() -> None:
    await mcp.run_stdio_async()


if __name__ == "__main__":
    asyncio.run(main())