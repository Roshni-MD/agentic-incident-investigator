from dataclasses import dataclass

from .models import InvestigationFinding


@dataclass
class FindingEvaluation:
    matches_expected_hypothesis: bool
    confidence: float
    evidence_count: int


def evaluate_finding(
    finding: InvestigationFinding,
    expected_hypothesis: str,
) -> FindingEvaluation:
    return FindingEvaluation(
        matches_expected_hypothesis=(
            finding.hypothesis == expected_hypothesis
        ),
        confidence=finding.confidence,
        evidence_count=len(finding.evidence),
    )