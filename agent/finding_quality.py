from .models import InvestigationFinding


def rank_findings(
    findings: list[InvestigationFinding],
) -> list[InvestigationFinding]:
    """Rank findings by confidence, highest first."""

    return sorted(
        findings,
        key=lambda finding: finding.confidence,
        reverse=True,
    )