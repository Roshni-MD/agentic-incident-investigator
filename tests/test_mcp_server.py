from datetime import timedelta

import pytest

from mcp_server.server import incident, mcp

@pytest.mark.asyncio
async def test_mcp_tools_are_registered():
    tools = await mcp.list_tools()

    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "get_current_metric",
        "get_metric_history",
        "get_service_health",
        "query_logs",
        "get_recent_deployments",
    }


@pytest.mark.asyncio
async def test_current_metric_tool_schema():
    tools = await mcp.list_tools()

    tool = next(
        tool
        for tool in tools
        if tool.name == "get_current_metric"
    )

    assert "metric_name" in tool.input_schema["properties"]
    assert "service_name" in tool.input_schema["properties"]

    assert set(tool.input_schema["required"]) == {
        "metric_name",
        "service_name",
    }

@pytest.mark.asyncio
async def test_get_current_metric_tool():
    result = await mcp.call_tool(
        "get_current_metric",
        {
            "metric_name": "gpu_utilization",
            "service_name": "image-ranking-service",
        },
    )

    assert result.is_error is False
    assert result.structured_content["service_name"] == "image-ranking-service"
    assert result.structured_content["metric_name"] == "gpu_utilization"
    assert result.structured_content["value"] == 42.0
    assert result.structured_content["status"] == "ok"

@pytest.mark.asyncio
async def test_get_service_health_tool():
    result = await mcp.call_tool(
        "get_service_health",
        {
            "service_name": "image-ranking-service",
        },
    )

    assert result.is_error is False

    data = result.structured_content

    assert data["service_name"] == "image-ranking-service"
    assert data["status"] == "ok"

    metrics = data["metrics"]

    assert metrics["cpu_utilization"]["value"] == 96.0
    assert metrics["gpu_utilization"]["value"] == 42.0
    assert metrics["gpu_memory_utilization"]["value"] == 70.0
    assert metrics["p95_latency_ms"]["value"] == 135.0
    assert metrics["throughput_rps"]["value"] == 760.0
    assert metrics["data_loading_ms"]["value"] == 380.0
    assert metrics["gpu_kernel_ms"]["value"] == 90.0

@pytest.mark.asyncio
async def test_query_logs_tool():
    result = await mcp.call_tool(
        "query_logs",
        {
            "service_name": incident.service_name,
            "start_time": incident.started_at.isoformat(),
            "end_time": incident.metrics[-1].timestamp.isoformat(),
        },
    )

    assert result.is_error is False

    data = result.structured_content["result"]

    assert len(data) == 2

    assert data[0]["service_name"] == "image-ranking-service"
    assert data[0]["level"] == "WARNING"
    assert data[0]["message"] == "Data preprocessing latency increased"
    assert data[0]["metadata"]["component"] == "image-preprocessor"
    assert data[0]["metadata"]["version"] == "v2.4"

    assert data[1]["level"] == "WARNING"
    assert data[1]["message"] == "GPU utilization below expected threshold"


@pytest.mark.asyncio
async def test_query_logs_empty_time_range():
    start = incident.metrics[-1].timestamp + timedelta(minutes=1)
    end = start + timedelta(minutes=1)

    result = await mcp.call_tool(
        "query_logs",
        {
            "service_name": incident.service_name,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
        },
    )

    assert result.is_error is False
    assert result.structured_content["result"] == []


@pytest.mark.asyncio
async def test_get_recent_deployments_tool():
    result = await mcp.call_tool(
        "get_recent_deployments",
        {
            "service_name": incident.service_name,
            "start_time": incident.metrics[0].timestamp.isoformat(),
            "end_time": incident.metrics[-1].timestamp.isoformat(),
        },
    )

    assert result.is_error is False

    data = result.structured_content["result"]

    assert len(data) == 1

    deployment = data[0]

    assert deployment["deployment_id"] == "deploy-1002"
    assert deployment["service_name"] == "image-ranking-service"
    assert deployment["model_name"] == "ranking-model"
    assert deployment["model_version"] == "v2.4"
    assert deployment["previous_version"] == "v2.3"

@pytest.mark.asyncio
async def test_get_recent_deployments_empty_time_range():
    start = incident.metrics[-1].timestamp + timedelta(minutes=1)
    end = start + timedelta(minutes=1)

    result = await mcp.call_tool(
        "get_recent_deployments",
        {
            "service_name": incident.service_name,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
        },
    )

    assert result.is_error is False
    assert result.structured_content["result"] == []