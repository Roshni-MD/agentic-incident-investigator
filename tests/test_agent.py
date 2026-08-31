import pytest

from agent.investigator import IncidentInvestigator
from agent.models import (
    AgentMessage,
    AgentResponse,
    AgentToolCall,
)
from agent.tools import AgentToolRegistry
from investigation.analyzer import IncidentAnalyzer
from telemetry.scenarios import load_cpu_bottleneck_scenario

from agent.llm import LLMClient
from agent.runner import AgentRunner


class MockLLM(LLMClient):

    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self,
        messages: list[AgentMessage],
        tools=None,
    ) -> AgentResponse:
        self.calls += 1

        if self.calls == 1:
            return AgentResponse(
                answer="",
                tool_calls=[
                    AgentToolCall(
                        tool_name="get_service_health",
                        arguments={
                            "service_name": "image-ranking-service",
                        },
                    )
                ],
            )

        return AgentResponse(
            answer="The service has a CPU-side bottleneck.",
        )


class InfiniteToolLLM(LLMClient):

    async def generate(
        self,
        messages: list[AgentMessage],
        tools=None,
    ) -> AgentResponse:
        return AgentResponse(
            answer="",
            tool_calls=[
                AgentToolCall(
                    tool_name="get_service_health",
                    arguments={
                        "service_name": "image-ranking-service",
                    },
                )
            ],
        )

def test_agent_message():
    message = AgentMessage(
        role="user",
        content="Investigate this incident.",
    )

    assert message.role == "user"
    assert message.content == "Investigate this incident."


def test_agent_tool_call():
    tool_call = AgentToolCall(
        tool_name="get_service_health",
        arguments={
            "service_name": "image-ranking-service",
        },
    )

    assert tool_call.tool_name == "get_service_health"
    assert tool_call.arguments["service_name"] == "image-ranking-service"


def test_agent_response():
    response = AgentResponse(
        answer="The service appears to have a CPU bottleneck.",
        tool_calls=[
            AgentToolCall(
                tool_name="get_service_health",
                arguments={
                    "service_name": "image-ranking-service",
                },
            )
        ],
    )

    assert (
        response.answer
        == "The service appears to have a CPU bottleneck."
    )

    assert len(response.tool_calls) == 1
    assert response.tool_calls[0].tool_name == "get_service_health"


def test_investigation_context():
    incident, _ = load_cpu_bottleneck_scenario()

    from agent.context import build_investigation_context

    context = build_investigation_context(incident)

    assert context.incident_id == incident.incident_id
    assert context.service_name == incident.service_name
    assert context.incident_type == incident.incident_type.value
    assert context.started_at == incident.started_at.isoformat()

    assert set(context.available_tools) == {
        "get_current_metric",
        "get_metric_history",
        "get_service_health",
        "query_logs",
        "get_recent_deployments",
    }


@pytest.mark.asyncio
async def test_tool_registry():
    registry = AgentToolRegistry()

    async def test_tool(service_name: str) -> dict[str, str]:
        return {"service_name": service_name}

    registry.register(
        "test_tool",
        test_tool,
    )

    assert "test_tool" in registry.names()

    tool = registry.get("test_tool")

    result = await tool(
        service_name="image-ranking-service",
    )

    assert result == {
        "service_name": "image-ranking-service",
    }


def test_tool_registry_multiple_tools():
    registry = AgentToolRegistry()

    async def tool_one() -> str:
        return "one"

    async def tool_two() -> str:
        return "two"

    registry.register("tool_one", tool_one)
    registry.register("tool_two", tool_two)

    assert set(registry.names()) == {
        "tool_one",
        "tool_two",
    }


def test_incident_investigator():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)
    investigator = IncidentInvestigator(analyzer)

    report = investigator.investigate(incident)

    assert report.incident_id == incident.incident_id
    assert report.service_name == incident.service_name

    assert report.likely_root_cause == (
        "CPU-side preprocessing bottleneck"
    )

    assert report.confidence == 0.87
    assert len(report.hypotheses) >= 1
    assert len(report.evidence) >= 1

@pytest.mark.asyncio
async def test_agent_runner_executes_tool_calls():
    incident, _ = load_cpu_bottleneck_scenario()

    registry = AgentToolRegistry()

    calls = []

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        calls.append(service_name)

        return {
            "service_name": service_name,
            "status": "ok",
        }

    registry.register(
        "get_service_health",
        get_service_health,
    )

    llm = MockLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == (
        "The service has a CPU-side bottleneck."
    )

    assert calls == [
        "image-ranking-service",
    ]

    assert llm.calls == 2

@pytest.mark.asyncio
async def test_agent_runner_enforces_iteration_limit():
    incident, _ = load_cpu_bottleneck_scenario()

    registry = AgentToolRegistry()

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        return {
            "service_name": service_name,
            "status": "ok",
        }

    registry.register(
        "get_service_health",
        get_service_health,
    )

    runner = AgentRunner(
        llm=InfiniteToolLLM(),
        tools=registry,
        max_iterations=3,
    )

    with pytest.raises(RuntimeError, match="maximum number"):
        await runner.run(incident)