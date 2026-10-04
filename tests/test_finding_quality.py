from agent.finding_quality import rank_findings
from agent.models import InvestigationFinding


def test_rank_findings_by_confidence():
    findings = [
        InvestigationFinding(
            hypothesis="Network bottleneck",
            confidence=0.61,
        ),
        InvestigationFinding(
            hypothesis="CPU-side preprocessing bottleneck",
            confidence=0.92,
        ),
        InvestigationFinding(
            hypothesis="GPU memory pressure",
            confidence=0.78,
        ),
    ]

    ranked = rank_findings(findings)

    assert [
        finding.hypothesis
        for finding in ranked
    ] == [
        "CPU-side preprocessing bottleneck",
        "GPU memory pressure",
        "Network bottleneck",
    ]

def test_rank_findings_does_not_mutate_input():
    first = InvestigationFinding(
        hypothesis="Network bottleneck",
        confidence=0.61,
    )

    second = InvestigationFinding(
        hypothesis="CPU-side preprocessing bottleneck",
        confidence=0.92,
    )

    findings = [first, second]

    ranked = rank_findings(findings)

    assert findings == [first, second]
    assert ranked == [second, first]