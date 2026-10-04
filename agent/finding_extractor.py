import json
from abc import ABC, abstractmethod
from typing import Any

from .llm import LLMClient
from .models import (
    AgentMessage,
    InvestigationEvidence,
    InvestigationFinding,
)


class FindingExtractor(ABC):
    """Interface for converting investigation evidence into findings."""

    @abstractmethod
    async def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        """Extract structured findings from investigation evidence."""
        raise NotImplementedError


class EvidenceFindingExtractor(FindingExtractor):
    """
    Deterministic fallback extractor.

    Useful for tests and environments where an LLM should not be called.
    """

    async def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        if not evidence:
            return []

        return [
            InvestigationFinding(
                hypothesis="Evidence requires further analysis",
                confidence=0.0,
                explanation=(
                    "The investigation collected telemetry evidence, "
                    "but no root-cause hypothesis has been established yet."
                ),
                evidence=evidence,
            )
        ]


class LLMFindingExtractor(FindingExtractor):
    """Use an LLM to convert structured evidence into an investigation finding."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        if not evidence:
            return []

        prompt = self._build_prompt(evidence)

        response = await self.llm.generate(
            messages=[
                AgentMessage(
                    role="system",
                    content=(
                        "You are an ML infrastructure incident "
                        "investigation analyst."
                    ),
                ),
                AgentMessage(
                    role="user",
                    content=prompt,
                ),
            ],
            tools=None,
        )

        finding = self._parse_finding(
            answer=response.answer,
            evidence=evidence,
        )

        return [finding]

    def _build_prompt(
        self,
        evidence: list[InvestigationEvidence],
    ) -> str:
        evidence_lines = []

        for index, item in enumerate(evidence):
            evidence_lines.append(
                f"[{index}] "
                f"source={item.source}; "
                f"description={item.description}; "
                f"value={json.dumps(item.value, default=str)}"
            )

        evidence_text = "\n".join(evidence_lines)

        return f"""
Analyze the collected telemetry evidence and identify the most likely
root cause of the incident.

Return ONLY valid JSON using this schema:

{{
  "hypothesis": "string",
  "confidence": 0.0,
  "explanation": "string",
  "evidence_indices": [0, 1],
  "recommended_actions": ["string"]
}}

Requirements:

- confidence must be a number between 0 and 1.
- evidence_indices must contain only indices from the evidence below.
- Use evidence_indices to identify the evidence supporting your hypothesis.
- Do not invent telemetry values.
- Do not reference evidence that is not present below.
- recommended_actions must contain concrete investigation or remediation actions.
- Return JSON only. Do not include markdown or additional commentary.

Collected evidence:

{evidence_text}
""".strip()

    def _parse_finding(
        self,
        answer: str,
        evidence: list[InvestigationEvidence],
    ) -> InvestigationFinding:
        try:
            data = json.loads(answer)

            if not isinstance(data, dict):
                return self._fallback_finding(evidence)

            hypothesis = data.get("hypothesis")
            confidence = data.get("confidence")
            explanation = data.get("explanation", "")
            evidence_indices = data.get("evidence_indices")
            recommended_actions = data.get(
                "recommended_actions",
                [],
            )

            if not isinstance(hypothesis, str) or not hypothesis.strip():
                return self._fallback_finding(evidence)

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= confidence <= 1
            ):
                return self._fallback_finding(evidence)

            if not isinstance(explanation, str):
                return self._fallback_finding(evidence)

            if not isinstance(evidence_indices, list):
                return self._fallback_finding(evidence)

            selected_evidence = []

            for index in evidence_indices:
                if (
                    isinstance(index, bool)
                    or not isinstance(index, int)
                    or index < 0
                    or index >= len(evidence)
                ):
                    return self._fallback_finding(evidence)

                selected_evidence.append(evidence[index])

            if not isinstance(recommended_actions, list):
                return self._fallback_finding(evidence)

            if not all(
                isinstance(action, str)
                for action in recommended_actions
            ):
                return self._fallback_finding(evidence)

            return InvestigationFinding(
                hypothesis=hypothesis,
                confidence=float(confidence),
                explanation=explanation,
                evidence=selected_evidence,
                recommended_actions=recommended_actions,
            )

        except (json.JSONDecodeError, TypeError, ValueError):
            return self._fallback_finding(evidence)

    def _fallback_finding(
        self,
        evidence: list[InvestigationEvidence],
    ) -> InvestigationFinding:
        return InvestigationFinding(
            hypothesis="Unable to determine root cause",
            confidence=0.0,
            explanation=(
                "The investigation collected telemetry evidence, "
                "but the finding model returned an invalid "
                "structured response."
            ),
            evidence=evidence,
            recommended_actions=[
                "Review the collected telemetry evidence manually."
            ],
        )