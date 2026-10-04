import json

from telemetry.models import Incident

from .context import build_investigation_context, build_system_prompt
from .llm import LLMClient
from .models import AgentMessage
from .state import AgentRunResult, AgentState
from .tool_executor import ToolExecutor
from .tools import AgentToolRegistry
from .evidence import EvidenceCollector
from agent import state
from .finding_extractor import FindingExtractor, LLMFindingExtractor


class AgentRunner:
    """Runs the investigation agent tool-calling loop."""

    def __init__(
        self,
        llm: LLMClient,
        tools: AgentToolRegistry,
        max_iterations: int = 10,
        evidence_collector: EvidenceCollector | None = None,
        finding_extractor: FindingExtractor | None = None,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.tool_executor = ToolExecutor(tools)
        self.max_iterations = max_iterations
        self.evidence_collector = (
            evidence_collector or EvidenceCollector()
        )
        self.finding_extractor = (
            finding_extractor
            if finding_extractor is not None
            else LLMFindingExtractor(llm)
        )

    async def run(
        self,
        incident: Incident,
    ) -> AgentRunResult:

        context = build_investigation_context(
            incident=incident,
            tools=self.tools,
        )

        state = AgentState(
            messages = [
                AgentMessage(
                    role="system",
                    content=build_system_prompt(context),
                ),
                AgentMessage(
                    role="user",
                    content=(
                        f"Investigate incident {incident.incident_id} "
                        f"for service {incident.service_name}."
                    ),
                ),
            ]
        )

        for _ in range(self.max_iterations):
            state.next_iteration()

            response = await self.llm.generate(
                state.messages,
                tools=self.tools.schemas(),
            )

            if not response.tool_calls:
                state.mark_completed()

                findings = await self.finding_extractor.extract(
                    state.evidence
                )

                return AgentRunResult(
                    answer=response.answer,
                    state=state,
                    findings=findings,
                )

            state.add_assistant_response(response)

            for tool_call in response.tool_calls:
                result = await self.tool_executor.execute(tool_call)

                evidence = self.evidence_collector.collect(
                    tool_name=tool_call.tool_name,
                    result=result,
                )

                for item in evidence:
                    state.add_evidence(item)

                state.add_tool_result(
                    content=json.dumps(result),
                    tool_call_id=tool_call.tool_call_id,
                )

        state.mark_failed()
        raise RuntimeError(
            "Agent exceeded maximum number of iterations."
        )