import json
import pytest

from agent import evidence
from agent.investigator import IncidentInvestigator
from agent.models import (
    AgentMessage,
    AgentResponse,
    AgentToolCall,
    InvestigationEvidence,
    InvestigationFinding,
)
from agent.tools import AgentToolRegistry
from investigation.analyzer import IncidentAnalyzer
from telemetry.scenarios import load_cpu_bottleneck_scenario

from agent.llm import LLMClient
from agent.runner import AgentRunner
from agent.telemetry_tools import build_telemetry_tool_registry
from agent.context import (
    build_investigation_context,
    build_system_prompt,
)
from agent.tool_executor import ToolExecutor
from agent.state import AgentRunResult, AgentState, InvestigationStatus
from agent.finding_extractor import FindingExtractor


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


class TestFindingExtractor(FindingExtractor):
    def extract(self, evidence):
        return [
            InvestigationFinding(
                hypothesis="Test hypothesis",
                confidence=0.75,
                explanation="Test explanation",
                evidence=evidence,
                recommended_actions=["Test action"],
            )
        ]


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
    incident, repository = load_cpu_bottleneck_scenario()
    registry = build_telemetry_tool_registry(repository)

    context = build_investigation_context(
        incident=incident,
        tools=registry,
    )

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
    assert response.state.status == InvestigationStatus.COMPLETED

    assert len(response.state.evidence) == 1

    evidence = response.state.evidence[0]

    assert evidence.source == "get_service_health"
    assert evidence.description == "Result returned by get_service_health"
    assert evidence.value is not None

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

def test_tool_registry_generates_openai_schemas():
    registry = AgentToolRegistry()

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        """Get the current health of an ML service."""
        return {
            "service_name": service_name,
            "status": "ok",
        }

    registry.register(
        "get_service_health",
        get_service_health,
    )

    schemas = registry.schemas()

    assert len(schemas) == 1

    schema = schemas[0]

    assert schema["type"] == "function"

    function = schema["function"]

    assert function["name"] == "get_service_health"
    assert (
        function["description"]
        == "Get the current health of an ML service."
    )

    assert (
        function["parameters"]["properties"]["service_name"]["type"]
        == "string"
    )

    assert function["parameters"]["required"] == [
        "service_name",
    ]

@pytest.mark.asyncio
async def test_agent_runner_supports_multi_step_investigation():
    incident, _ = load_cpu_bottleneck_scenario()

    registry = AgentToolRegistry()

    calls: list[str] = []

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        calls.append("get_service_health")

        return {
            "service_name": service_name,
            "status": "degraded",
            "cpu_utilization": 96.0,
            "gpu_utilization": 42.0,
        }

    async def query_logs(
        service_name: str,
    ) -> dict[str, object]:
        calls.append("query_logs")

        return [
            {
                "timestamp": "2026-08-29T16:58:00+00:00",
                "service_name": service_name,
                "level": "WARNING",
                "message": "Data preprocessing latency increased",
                "metadata": {},
            }
        ]

    registry.register(
        "get_service_health",
        get_service_health,
    )

    registry.register(
        "query_logs",
        query_logs,
    )

    class MultiStepLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            if self.call_count == 1:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_service_health",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_health",
                        )
                    ]
                )

            if self.call_count == 2:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="query_logs",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_logs",
                        )
                    ]
                )

            return AgentResponse(
                answer=(
                    "The incident is likely caused by a "
                    "CPU-side preprocessing bottleneck."
                )
            )

    llm = MultiStepLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == (
        "The incident is likely caused by a "
        "CPU-side preprocessing bottleneck."
    )
    assert response.state.status == InvestigationStatus.COMPLETED
    assert response.state.iteration == 3
    assert response.findings
    assert len(response.findings) == 1
    finding = response.findings[0]

    assert finding.hypothesis == "Evidence requires further analysis"
    assert finding.confidence == 0.0
    assert finding.evidence
    assert len(finding.evidence) == len(response.state.evidence)

    assert calls == [
        "get_service_health",
        "query_logs",
    ]

    assert llm.call_count == 3

@pytest.mark.asyncio
async def test_telemetry_tool_registry_uses_real_repository():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    health_tool = registry.get("get_service_health")

    result = await health_tool(
        service_name=incident.service_name,
    )

    assert result["service_name"] == incident.service_name
    assert result["status"] == "ok"

    metrics = result["metrics"]

    assert metrics["cpu_utilization"]["value"] == 96.0
    assert metrics["gpu_utilization"]["value"] == 42.0
    assert metrics["gpu_memory_utilization"]["value"] == 70.0

@pytest.mark.asyncio
async def test_agent_runner_passes_tool_schemas_to_llm():
    incident, _ = load_cpu_bottleneck_scenario()

    registry = AgentToolRegistry()

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        """Get the current health of an ML service."""
        return {
            "service_name": service_name,
            "status": "ok",
        }

    registry.register(
        "get_service_health",
        get_service_health,
    )

    class SchemaCapturingLLM(LLMClient):
        def __init__(self) -> None:
            self.received_tools = None

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.received_tools = tools

            return AgentResponse(
                answer="Investigation complete.",
            )

    llm = SchemaCapturingLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == "Investigation complete."
    assert response.state.status == InvestigationStatus.COMPLETED

    assert llm.received_tools is not None
    assert len(llm.received_tools) == 1

    schema = llm.received_tools[0]

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "get_service_health"

"""
This is the first test that connects the pieces together:

load_cpu_bottleneck_scenario()
             │
             ▼
TelemetryRepository
             │
             ▼
build_telemetry_tool_registry()
             │
             ▼
AgentToolRegistry
             │
             ▼
AgentRunner
             │
             ├── sends tool schemas ──────► Mock LLM
             │
             ◄── get_service_health ────────┤
             │
             ▼
actual telemetry tool
             │
             ▼
TelemetryRepository
             │
             ▼
tool result
             │
             ▼
AgentRunner ───────────────────────────────► Mock LLM
                                             │
                                             ▼
                                        final answer
"""
@pytest.mark.asyncio
async def test_agent_runner_uses_real_telemetry_tools():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    class RealTelemetryLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0
            self.received_tool_results = []

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            if self.call_count == 1:
                # Verify the real telemetry tools were exposed to the LLM.
                assert tools is not None

                tool_names = {
                    tool["function"]["name"]
                    for tool in tools
                }

                assert tool_names == {
                    "get_current_metric",
                    "get_metric_history",
                    "get_service_health",
                    "query_logs",
                    "get_recent_deployments",
                }

                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_service_health",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_health",
                        )
                    ]
                )

            # Verify that the tool result was returned to the LLM.
            tool_messages = [
                message
                for message in messages
                if message.role == "tool"
            ]

            assert len(tool_messages) == 1
            assert tool_messages[0].tool_call_id == "call_health"
            assert "image-ranking-service" in tool_messages[0].content
            assert "cpu_utilization" in tool_messages[0].content

            return AgentResponse(
                answer="The service has a CPU-side bottleneck."
            )

    llm = RealTelemetryLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == (
        "The service has a CPU-side bottleneck."
    )
    assert response.state.status == InvestigationStatus.COMPLETED
    assert response.state.iteration == 2

    assert llm.call_count == 2

@pytest.mark.asyncio
async def test_telemetry_tool_registry_gets_metric_history():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    tool = registry.get("get_metric_history")

    start_time = incident.metrics[0].timestamp.isoformat()
    end_time = incident.metrics[-1].timestamp.isoformat()

    result = await tool(
        metric_name="cpu_utilization",
        service_name=incident.service_name,
        start_time=start_time,
        end_time=end_time,
    )

    assert len(result) > 0
    assert "timestamp" in result[0]
    assert "value" in result[0]


@pytest.mark.asyncio
async def test_telemetry_tool_registry_queries_logs():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    tool = registry.get("query_logs")

    result = await tool(
        service_name=incident.service_name,
        start_time=incident.metrics[0].timestamp.isoformat(),
        end_time=incident.metrics[-1].timestamp.isoformat(),
    )

    assert len(result) > 0
    assert result[0]["service_name"] == incident.service_name
    assert "level" in result[0]
    assert "message" in result[0]


@pytest.mark.asyncio
async def test_telemetry_tool_registry_gets_recent_deployments():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    tool = registry.get("get_recent_deployments")

    result = await tool(
        service_name=incident.service_name,
        start_time=incident.metrics[0].timestamp.isoformat(),
        end_time=incident.metrics[-1].timestamp.isoformat(),
    )

    assert len(result) > 0
    assert result[0]["service_name"] == incident.service_name
    assert "deployment_id" in result[0]
    assert "model_version" in result[0]

@pytest.mark.asyncio
async def test_agent_runner_performs_multi_source_investigation():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    class MultiSourceLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0
            self.tool_results = []

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            tool_messages = [
                message
                for message in messages
                if message.role == "tool"
            ]

            self.tool_results = tool_messages

            if self.call_count == 1:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_service_health",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_health",
                        )
                    ]
                )

            if self.call_count == 2:
                assert len(tool_messages) == 1

                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_metric_history",
                            arguments={
                                "metric_name": "cpu_utilization",
                                "service_name": incident.service_name,
                                "start_time": incident.metrics[0].timestamp.isoformat(),
                                "end_time": incident.metrics[-1].timestamp.isoformat(),
                            },
                            tool_call_id="call_history",
                        )
                    ]
                )

            if self.call_count == 3:
                assert len(tool_messages) == 2

                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="query_logs",
                            arguments={
                                "service_name": incident.service_name,
                                "start_time": incident.metrics[0].timestamp.isoformat(),
                                "end_time": incident.metrics[-1].timestamp.isoformat(),
                            },
                            tool_call_id="call_logs",
                        )
                    ]
                )

            if self.call_count == 4:
                assert len(tool_messages) == 3

                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_recent_deployments",
                            arguments={
                                "service_name": incident.service_name,
                                "start_time": incident.metrics[0].timestamp.isoformat(),
                                "end_time": incident.metrics[-1].timestamp.isoformat(),
                            },
                            tool_call_id="call_deployments",
                        )
                    ]
                )

            assert self.call_count == 5
            assert len(tool_messages) == 4

            return AgentResponse(
                answer=(
                    "The service has a CPU-side bottleneck. "
                    "Telemetry shows elevated CPU utilization and "
                    "the supporting investigation evidence is consistent "
                    "with CPU-side preprocessing."
                )
            )

    llm = MultiSourceLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer.startswith(
        "The service has a CPU-side bottleneck."
    )
    assert response.state.status == InvestigationStatus.COMPLETED
    assert response.state.iteration == 5

    assert llm.call_count == 5
    assert len(llm.tool_results) == 4

@pytest.mark.asyncio
async def test_agent_runner_serializes_tool_results_as_json():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    class JsonToolResultLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            if self.call_count == 1:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_service_health",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_health",
                        )
                    ]
                )

            tool_messages = [
                message
                for message in messages
                if message.role == "tool"
            ]

            assert len(tool_messages) == 1

            result = json.loads(tool_messages[0].content)

            assert result["service_name"] == incident.service_name
            assert result["status"] == "ok"
            assert "metrics" in result

            return AgentResponse(
                answer="Tool result was valid JSON."
            )

    llm = JsonToolResultLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == "Tool result was valid JSON."
    assert response.state.status == InvestigationStatus.COMPLETED

@pytest.mark.asyncio
async def test_agent_runner_provides_investigation_context():
    incident, repository = load_cpu_bottleneck_scenario()

    registry = build_telemetry_tool_registry(repository)

    class ContextAwareLLM(LLMClient):
        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            system_message = messages[0]

            assert system_message.role == "system"
            assert incident.incident_id in system_message.content
            assert incident.service_name in system_message.content
            assert incident.incident_type.value in system_message.content
            assert incident.started_at.isoformat() in system_message.content

            for tool_name in registry.names():
                assert tool_name in system_message.content

            return AgentResponse(
                answer="Context received successfully."
            )

    llm = ContextAwareLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
    )

    response = await runner.run(incident)

    assert response.answer == "Context received successfully."
    assert response.state.status == InvestigationStatus.COMPLETED

def test_build_investigation_context():
    incident, repository = load_cpu_bottleneck_scenario()
    registry = build_telemetry_tool_registry(repository)

    context = build_investigation_context(
        incident=incident,
        tools=registry,
    )

    assert context.incident_id == incident.incident_id
    assert context.service_name == incident.service_name
    assert context.incident_type == incident.incident_type.value
    assert context.started_at == incident.started_at.isoformat()
    assert set(context.available_tools) == set(registry.names())

def test_build_system_prompt_contains_investigation_context():
    incident, repository = load_cpu_bottleneck_scenario()
    registry = build_telemetry_tool_registry(repository)

    context = build_investigation_context(
        incident=incident,
        tools=registry,
    )

    prompt = build_system_prompt(context)

    assert "ML infrastructure incident investigation agent" in prompt
    assert incident.incident_id in prompt
    assert incident.service_name in prompt
    assert incident.incident_type.value in prompt
    assert incident.started_at.isoformat() in prompt

    for tool_name in registry.names():
        assert tool_name in prompt

@pytest.mark.asyncio
async def test_tool_executor_executes_registered_tool():
    incident, repository = load_cpu_bottleneck_scenario()
    registry = build_telemetry_tool_registry(repository)

    executor = ToolExecutor(registry)

    tool_call = AgentToolCall(
        tool_name="get_current_metric",
        arguments={
            "metric_name": "gpu_utilization",
            "service_name": incident.service_name,
        },
        tool_call_id="call_gpu",
    )

    result = await executor.execute(tool_call)

    assert result["service_name"] == incident.service_name
    assert result["metric_name"] == "gpu_utilization"
    assert result["status"] == "ok"
    assert "value" in result
    assert "timestamp" in result

def test_agent_state_starts_empty():
    state = AgentState()

    assert state.messages == []
    assert state.iteration == 0

@pytest.mark.asyncio
async def test_agent_runner_tracks_investigation_state():
    incident, repository = load_cpu_bottleneck_scenario()
    registry = build_telemetry_tool_registry(repository)

    captured_state = {}

    class StateTrackingLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            captured_state["message_count"] = len(messages)

            if self.call_count == 1:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_current_metric",
                            arguments={
                                "metric_name": "cpu_utilization",
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call_cpu",
                        )
                    ]
                )

            assert self.call_count == 2

            tool_messages = [
                message for message in messages
                if message.role == "tool"
            ]

            assert len(tool_messages) == 1
            assert messages[0].role == "system"
            assert messages[1].role == "user"
            assert messages[2].role == "assistant"
            assert messages[3].role == "tool"

            return AgentResponse(
                answer="Investigation complete."
            )

    llm = StateTrackingLLM()
    runner = AgentRunner(llm=llm, tools=registry)

    response = await runner.run(incident)

    assert response.answer == "Investigation complete."
    assert response.state.status == InvestigationStatus.COMPLETED
    assert response.state.iteration == 2
    assert llm.call_count == 2
    assert captured_state["message_count"] == 4
    assert response.findings
    assert len(response.findings) == 1
    assert response.findings[0].hypothesis == "Evidence requires further analysis"
    assert len(response.findings[0].evidence) > 0

def test_agent_state_adds_messages():
    state = AgentState()

    message = AgentMessage(
        role="user",
        content="Investigate the incident.",
    )

    state.add_message(message)

    assert state.messages == [message]

def test_agent_state_adds_assistant_response():
    state = AgentState()

    response = AgentResponse(
        answer="I found elevated CPU utilization.",
    )

    state.add_assistant_response(response)

    assert len(state.messages) == 1
    assert state.messages[0].role == "assistant"
    assert state.messages[0].content == response.answer
    assert state.messages[0].tool_calls == []

def test_agent_state_adds_tool_result():
    state = AgentState()

    state.add_tool_result(
        content='{"status": "ok"}',
        tool_call_id="call_123",
    )

    assert len(state.messages) == 1
    assert state.messages[0].role == "tool"
    assert state.messages[0].content == '{"status": "ok"}'
    assert state.messages[0].tool_call_id == "call_123"

def test_agent_state_advances_iteration():
    state = AgentState()

    assert state.iteration == 0

    state.next_iteration()
    assert state.iteration == 1

    state.next_iteration()
    assert state.iteration == 2

def test_agent_state_starts_running():
    state = AgentState()

    assert state.status == InvestigationStatus.RUNNING

def test_agent_state_can_be_completed():
    state = AgentState()

    state.mark_completed()

    assert state.status == InvestigationStatus.COMPLETED

def test_agent_state_can_fail():
    state = AgentState()

    state.mark_failed()

    assert state.status == InvestigationStatus.FAILED

def test_agent_run_result_contains_answer_and_state():
    state = AgentState()
    state.mark_completed()

    result = AgentRunResult(
        answer="Investigation complete.",
        state=state,
    )

    assert result.answer == "Investigation complete."
    assert result.state is state
    assert result.state.status == InvestigationStatus.COMPLETED

def test_investigation_evidence():
    evidence = InvestigationEvidence(
        source="get_current_metric",
        description="CPU utilization is high",
        value=96,
    )

    assert evidence.source == "get_current_metric"
    assert evidence.description == "CPU utilization is high"
    assert evidence.value == 96


def test_investigation_finding():
    evidence = InvestigationEvidence(
        source="get_current_metric",
        description="CPU utilization is high",
        value=96,
    )

    finding = InvestigationFinding(
        hypothesis="CPU-side preprocessing bottleneck",
        confidence=0.92,
        evidence=[evidence],
    )

    assert finding.hypothesis == "CPU-side preprocessing bottleneck"
    assert finding.confidence == 0.92
    assert len(finding.evidence) == 1
    assert finding.evidence[0].value == 96


def test_agent_run_result_contains_findings():
    state = AgentState()
    state.mark_completed()

    evidence = InvestigationEvidence(
        source="get_current_metric",
        description="CPU utilization is high",
        value=96,
    )

    finding = InvestigationFinding(
        hypothesis="CPU-side preprocessing bottleneck",
        confidence=0.92,
        evidence=[evidence],
    )

    result = AgentRunResult(
        answer="CPU-side preprocessing is the likely bottleneck.",
        state=state,
        findings=[finding],
    )

    assert result.answer == "CPU-side preprocessing is the likely bottleneck."
    assert result.state is state
    assert result.findings == [finding]
    assert result.findings[0].confidence == 0.92

def test_agent_state_add_evidence():
    state = AgentState()

    evidence = InvestigationEvidence(
        source="get_current_metric",
        description="CPU utilization is high",
        value=96,
    )

    state.add_evidence(evidence)

    assert len(state.evidence) == 1
    assert state.evidence[0] is evidence


@pytest.mark.asyncio
async def test_agent_runner_uses_injected_finding_extractor():
    incident, _ = load_cpu_bottleneck_scenario()

    registry = AgentToolRegistry()

    async def get_service_health(
        service_name: str,
    ) -> dict[str, object]:
        """Get the current health of an ML service."""
        return {
            "service_name": service_name,
            "status": "ok",
            "cpu": 96,
            "gpu": 42,
        }

    registry.register(
        "get_service_health",
        get_service_health,
    )

    class TestLLM(LLMClient):
        def __init__(self) -> None:
            self.call_count = 0

        async def generate(
            self,
            messages: list[AgentMessage],
            tools=None,
        ) -> AgentResponse:
            self.call_count += 1

            if self.call_count == 1:
                return AgentResponse(
                    tool_calls=[
                        AgentToolCall(
                            tool_name="get_service_health",
                            arguments={
                                "service_name": incident.service_name,
                            },
                            tool_call_id="call-1",
                        )
                    ]
                )

            return AgentResponse(
                answer="Investigation complete.",
            )

    llm = TestLLM()

    runner = AgentRunner(
        llm=llm,
        tools=registry,
        finding_extractor=TestFindingExtractor(),
    )

    result = await runner.run(incident)

    assert result.answer == "Investigation complete."
    assert result.state.status == InvestigationStatus.COMPLETED

    # Verify evidence was actually collected.
    assert result.state.evidence
    assert len(result.state.evidence) > 0

    # Verify the injected extractor was used.
    assert result.findings
    assert len(result.findings) == 1

    finding = result.findings[0]

    assert finding.hypothesis == "Test hypothesis"
    assert finding.confidence == 0.75
    assert finding.explanation == "Test explanation"
    assert finding.recommended_actions == ["Test action"]

    # Verify the extractor received the evidence collected by the agent.
    assert finding.evidence == result.state.evidence