import json

import pytest

from agent.finding_extractor import LLMFindingExtractor
from agent.llm import LLMClient
from agent.models import (
    AgentMessage,
    AgentResponse,
    InvestigationEvidence,
)


class FakeFindingLLM(LLMClient):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages = []

    async def generate(
        self,
        messages: list[AgentMessage],
        tools=None,
    ) -> AgentResponse:
        self.messages = messages

        return AgentResponse(
            answer=self.answer,
        )


def build_test_evidence() -> list[InvestigationEvidence]:
    return [
        InvestigationEvidence(
            source="get_current_metric",
            description="cpu_utilization is 96",
            value=96,
        ),
        InvestigationEvidence(
            source="get_current_metric",
            description="gpu_utilization is 42",
            value=42,
        ),
        InvestigationEvidence(
            source="get_metric_history",
            description="Metric value is 380 at 2026-01-01T00:00:00",
            value=380,
        ),
    ]


@pytest.mark.asyncio
async def test_llm_finding_extractor_generates_structured_finding():
    answer = json.dumps(
        {
            "hypothesis": "CPU-side preprocessing bottleneck",
            "confidence": 0.92,
            "explanation": (
                "CPU utilization is high while GPU utilization "
                "is relatively low."
            ),
            "evidence_indices": [0, 1, 2],
            "recommended_actions": [
                "Investigate CPU-side preprocessing.",
                "Review the data-loading pipeline.",
            ],
        }
    )

    llm = FakeFindingLLM(answer)
    extractor = LLMFindingExtractor(llm)

    evidence = build_test_evidence()

    findings = await extractor.extract(evidence)

    assert len(findings) == 1

    finding = findings[0]

    assert finding.hypothesis == "CPU-side preprocessing bottleneck"
    assert finding.confidence == 0.92
    assert (
        finding.explanation
        == "CPU utilization is high while GPU utilization is relatively low."
    )

    assert finding.recommended_actions == [
        "Investigate CPU-side preprocessing.",
        "Review the data-loading pipeline.",
    ]

    assert finding.evidence == evidence


@pytest.mark.asyncio
async def test_llm_finding_extractor_maps_evidence_indices():
    answer = json.dumps(
        {
            "hypothesis": "CPU bottleneck",
            "confidence": 0.85,
            "explanation": "CPU is saturated.",
            "evidence_indices": [0, 2],
            "recommended_actions": [],
        }
    )

    llm = FakeFindingLLM(answer)
    extractor = LLMFindingExtractor(llm)

    evidence = build_test_evidence()

    findings = await extractor.extract(evidence)

    finding = findings[0]

    assert finding.evidence == [
        evidence[0],
        evidence[2],
    ]


@pytest.mark.asyncio
async def test_llm_finding_extractor_includes_evidence_in_prompt():
    answer = json.dumps(
        {
            "hypothesis": "CPU bottleneck",
            "confidence": 0.8,
            "explanation": "CPU is high.",
            "evidence_indices": [0],
            "recommended_actions": [],
        }
    )

    llm = FakeFindingLLM(answer)
    extractor = LLMFindingExtractor(llm)

    evidence = build_test_evidence()

    await extractor.extract(evidence)

    assert len(llm.messages) == 2

    system_message = llm.messages[0]
    user_message = llm.messages[1]

    assert system_message.role == "system"
    assert "ML infrastructure incident" in system_message.content

    assert user_message.role == "user"
    assert "[0]" in user_message.content
    assert "[1]" in user_message.content
    assert "[2]" in user_message.content

    assert "cpu_utilization is 96" in user_message.content
    assert "gpu_utilization is 42" in user_message.content
    assert "380" in user_message.content


@pytest.mark.asyncio
async def test_llm_finding_extractor_rejects_invalid_evidence_index():
    answer = json.dumps(
        {
            "hypothesis": "CPU bottleneck",
            "confidence": 0.8,
            "explanation": "CPU is high.",
            "evidence_indices": [99],
            "recommended_actions": [],
        }
    )

    llm = FakeFindingLLM(answer)
    extractor = LLMFindingExtractor(llm)

    evidence = build_test_evidence()

    findings = await extractor.extract(evidence)

    finding = findings[0]

    assert finding.hypothesis == "Unable to determine root cause"
    assert finding.confidence == 0.0
    assert finding.evidence == evidence


@pytest.mark.asyncio
async def test_llm_finding_extractor_returns_fallback_for_malformed_json():
    llm = FakeFindingLLM(
        "The CPU is probably the bottleneck."
    )

    extractor = LLMFindingExtractor(llm)

    evidence = build_test_evidence()

    findings = await extractor.extract(evidence)

    finding = findings[0]

    assert finding.hypothesis == "Unable to determine root cause"
    assert finding.confidence == 0.0
    assert finding.evidence == evidence
    assert finding.recommended_actions == [
        "Review the collected telemetry evidence manually."
    ]


@pytest.mark.asyncio
async def test_llm_finding_extractor_returns_empty_for_no_evidence():
    llm = FakeFindingLLM("{}")
    extractor = LLMFindingExtractor(llm)

    findings = await extractor.extract([])

    assert findings == []
    assert llm.messages == []