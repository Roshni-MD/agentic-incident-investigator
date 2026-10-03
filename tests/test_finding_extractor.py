from agent.finding_extractor import EvidenceFindingExtractor
from agent.models import InvestigationEvidence


def test_evidence_finding_extractor_returns_empty_for_no_evidence():
    extractor = EvidenceFindingExtractor()

    findings = extractor.extract([])

    assert findings == []


def test_evidence_finding_extractor_creates_finding_from_evidence():
    extractor = EvidenceFindingExtractor()

    evidence = [
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
    ]

    findings = extractor.extract(evidence)

    assert len(findings) == 1

    finding = findings[0]

    assert finding.hypothesis == "Evidence requires further analysis"
    assert finding.confidence == 0.0
    assert len(finding.evidence) == 2
    assert finding.evidence[0].description == "cpu_utilization is 96"