import pytest

from mcp_server.server import mcp


@pytest.mark.asyncio
async def test_mcp_tools_are_registered():
    tools = await mcp.list_tools()

    tool_names = {tool.name for tool in tools}

    assert tool_names == {
        "get_current_metric",
        "get_metric_history",
        "get_service_health",
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