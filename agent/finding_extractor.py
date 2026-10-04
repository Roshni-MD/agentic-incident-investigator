from abc import ABC, abstractmethod

from .llm import LLMClient
from .models import (
    AgentMessage,
    InvestigationEvidence,
    InvestigationFinding,
)


class FindingExtractor(ABC):
    """Extract structured findings from collected investigation evidence."""

    @abstractmethod
    async def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        """Return findings derived from investigation evidence."""
        raise NotImplementedError


class LLMFindingExtractor(FindingExtractor):
    """Use an LLM to analyze collected evidence and produce findings."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        if not evidence:
            return []

        evidence_text = "\n".join(
            f"- [{item.source}] {item.description}"
            for item in evidence
        )

        messages = [
            AgentMessage(
                role="system",
                content=(
                    "You are an ML infrastructure incident investigator. "
                    "Analyze the provided telemetry evidence and identify "
                    "the most likely root-cause hypothesis. "
                    "Return exactly one JSON object with these fields: "
                    "hypothesis, confidence, explanation, "
                    "evidence_indices, recommended_actions. "
                    "confidence must be a number between 0 and 1. "
                    "evidence_indices must contain zero-based indexes "
                    "of the supporting evidence items."
                ),
            ),
            AgentMessage(
                role="user",
                content=(
                    "Analyze the following incident evidence:\n\n"
                    f"{evidence_text}"
                ),
            ),
        ]

        response = await self.llm.generate(messages)

        return [
            self._parse_finding(
                response.answer,
                evidence,
            )
        ]

    def _parse_finding(
        self,
        answer: str,
        evidence: list[InvestigationEvidence],
    ) -> InvestigationFinding:
        import json

        data = json.loads(answer)

        evidence_indices = data.get("evidence_indices", [])

        selected_evidence = [
            evidence[index]
            for index in evidence_indices
            if isinstance(index, int) and 0 <= index < len(evidence)
        ]

        return InvestigationFinding(
            hypothesis=data["hypothesis"],
            confidence=data["confidence"],
            explanation=data.get("explanation", ""),
            evidence=selected_evidence,
            recommended_actions=data.get(
                "recommended_actions",
                [],
            ),
        )