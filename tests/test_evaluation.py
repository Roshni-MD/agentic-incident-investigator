from agent.evaluation import evaluate_finding
from agent.models import InvestigationEvidence, InvestigationFinding


def test_evaluate_finding_matches_expected_hypothesis():
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
    ]

    finding = InvestigationFinding(
        hypothesis="CPU-side preprocessing bottleneck",
        confidence=0.92,
        evidence=evidence,
    )

    evaluation = evaluate_finding(
        finding=finding,
        expected_hypothesis=(
            "CPU-side preprocessing bottleneck"
        ),
    )

    assert evaluation.matches_expected_hypothesis is True
    assert evaluation.confidence == 0.92
    assert evaluation.evidence_count == 2

def test_evaluate_finding_detects_mismatch():
    finding = InvestigationFinding(
        hypothesis="Network bottleneck",
        confidence=0.70,
    )

    evaluation = evaluate_finding(
        finding=finding,
        expected_hypothesis=(
            "CPU-side preprocessing bottleneck"
        ),
    )

    assert evaluation.matches_expected_hypothesis is False