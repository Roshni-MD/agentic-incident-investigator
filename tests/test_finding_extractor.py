import json
import pytest

from agent.finding_extractor import LLMFindingExtractor
from agent.models import AgentMessage, AgentResponse, InvestigationEvidence
from agent.llm import LLMClient


class MockFindingLLM(LLMClient):
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.messages = None

    async def generate(
        self,
        messages: list[AgentMessage],
        tools=None,
    ) -> AgentResponse:
        self.messages = messages

        return AgentResponse(
            answer=self.answer,
        )


@pytest.mark.asyncio
async def test_llm_finding_extractor_creates_structured_finding():
    llm = MockFindingLLM(
        """
        {
            "hypothesis": "CPU-side preprocessing bottleneck",
            "confidence": 0.87,
            "explanation": "CPU utilization is saturated while GPU utilization is low.",
            "evidence_indices": [0, 1],
            "recommended_actions": [
                "Profile CPU-side preprocessing",
                "Investigate data loading latency"
            ]
        }
        """
    )

    extractor = LLMFindingExtractor(llm)

    evidence = [
        InvestigationEvidence(
            source="get_service_health",
            description="cpu is 96",
            value=96,
        ),
        InvestigationEvidence(
            source="get_service_health",
            description="gpu is 42",
            value=42,
        ),
    ]

    findings = await extractor.extract(evidence)

    assert len(findings) == 1

    finding = findings[0]

    assert finding.hypothesis == "CPU-side preprocessing bottleneck"
    assert finding.confidence == 0.87
    assert (
        finding.explanation
        == "CPU utilization is saturated while GPU utilization is low."
    )

    assert len(finding.evidence) == 2
    assert finding.evidence[0] == evidence[0]
    assert finding.evidence[1] == evidence[1]

    assert finding.recommended_actions == [
        "Profile CPU-side preprocessing",
        "Investigate data loading latency",
    ]


@pytest.mark.asyncio
async def test_llm_finding_extractor_returns_empty_for_no_evidence():
    llm = MockFindingLLM("should not be called")

    extractor = LLMFindingExtractor(llm)

    findings = await extractor.extract([])

    assert findings == []
    assert llm.messages is None

@pytest.mark.asyncio
async def test_llm_finding_extractor_ignores_invalid_evidence_indices():
    llm = MockFindingLLM(
        """
        {
            "hypothesis": "CPU bottleneck",
            "confidence": 0.8,
            "explanation": "CPU utilization is high.",
            "evidence_indices": [0, 99, -1],
            "recommended_actions": []
        }
        """
    )

    extractor = LLMFindingExtractor(llm)

    evidence = [
        InvestigationEvidence(
            source="get_current_metric",
            description="cpu_utilization is 96",
            value=96,
        )
    ]

    findings = await extractor.extract(evidence)

    assert len(findings) == 1
    assert len(findings[0].evidence) == 1
    assert findings[0].evidence[0] == evidence[0]

@pytest.mark.asyncio
async def test_llm_finding_extractor_handles_invalid_json():
    class InvalidJsonLLM(LLMClient):
        async def generate(
            self,
            messages,
            tools=None,
        ):
            return AgentResponse(
                answer="This is not JSON."
            )

    evidence = [
        InvestigationEvidence(
            source="get_current_metric",
            description="CPU utilization is 96",
            value=96,
        )
    ]

    extractor = LLMFindingExtractor(
        InvalidJsonLLM()
    )

    findings = await extractor.extract(evidence)

    assert len(findings) == 1
    assert findings[0].hypothesis == (
        "Unable to determine root cause"
    )
    assert findings[0].confidence == 0.0
    assert findings[0].evidence == evidence

@pytest.mark.asyncio
async def test_llm_finding_extractor_handles_invalid_confidence():
    class InvalidConfidenceLLM(LLMClient):
        async def generate(
            self,
            messages,
            tools=None,
        ):
            return AgentResponse(
                answer=json.dumps(
                    {
                        "hypothesis": "CPU bottleneck",
                        "confidence": 2.5,
                        "explanation": "Invalid confidence",
                        "evidence_indices": [0],
                        "recommended_actions": [],
                    }
                )
            )

    evidence = [
        InvestigationEvidence(
            source="get_current_metric",
            description="CPU utilization is 96",
            value=96,
        )
    ]

    extractor = LLMFindingExtractor(
        InvalidConfidenceLLM()
    )

    findings = await extractor.extract(evidence)

    assert findings[0].hypothesis == (
        "Unable to determine root cause"
    )
    assert findings[0].confidence == 0.0

@pytest.mark.asyncio
async def test_llm_finding_extractor_handles_missing_hypothesis():
    class MissingHypothesisLLM(LLMClient):
        async def generate(
            self,
            messages,
            tools=None,
        ):
            return AgentResponse(
                answer=json.dumps(
                    {
                        "confidence": 0.9,
                        "explanation": "Something happened.",
                        "evidence_indices": [0],
                        "recommended_actions": [],
                    }
                )
            )

    evidence = [
        InvestigationEvidence(
            source="get_current_metric",
            description="CPU utilization is 96",
            value=96,
        )
    ]

    extractor = LLMFindingExtractor(
        MissingHypothesisLLM()
    )

    findings = await extractor.extract(evidence)

    assert findings[0].hypothesis == (
        "Unable to determine root cause"
    )

@pytest.mark.asyncio
async def test_llm_finding_extractor_preserves_evidence_traceability():
    class TraceabilityLLM(LLMClient):
        async def generate(
            self,
            messages,
            tools=None,
        ):
            return AgentResponse(
                answer=json.dumps(
                    {
                        "hypothesis": (
                            "CPU-side preprocessing bottleneck"
                        ),
                        "confidence": 0.92,
                        "explanation": (
                            "CPU utilization increased while "
                            "GPU utilization decreased."
                        ),
                        "evidence_indices": [0, 2],
                        "recommended_actions": [
                            "Investigate CPU-side preprocessing."
                        ],
                    }
                )
            )

    evidence = [
        InvestigationEvidence(
            source="get_current_metric",
            description="CPU utilization is 96",
            value=96,
        ),
        InvestigationEvidence(
            source="get_current_metric",
            description="GPU utilization is 42",
            value=42,
        ),
        InvestigationEvidence(
            source="get_metric_history",
            description="CPU increased from 20 to 96",
            value=96,
        ),
    ]

    extractor = LLMFindingExtractor(TraceabilityLLM())

    findings = await extractor.extract(evidence)

    finding = findings[0]

    assert finding.hypothesis == (
        "CPU-side preprocessing bottleneck"
    )

    assert finding.confidence == 0.92

    assert len(finding.evidence) == 2

    assert finding.evidence[0] is evidence[0]
    assert finding.evidence[1] is evidence[2]