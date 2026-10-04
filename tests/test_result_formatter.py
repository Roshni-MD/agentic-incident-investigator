from agent.models import InvestigationEvidence, InvestigationFinding
from agent.result_formatter import InvestigationResultFormatter
from agent.state import AgentRunResult, AgentState


def test_formats_finding_with_evidence_and_actions():
    finding = InvestigationFinding(
        hypothesis="CPU-side preprocessing bottleneck",
        confidence=0.91,
        explanation=(
            "CPU utilization is high while GPU utilization remains low."
        ),
        evidence=[
            InvestigationEvidence(
                source="get_service_health",
                description="cpu_utilization is 96.0",
                value=96.0,
            ),
            InvestigationEvidence(
                source="get_service_health",
                description="gpu_utilization is 42.0",
                value=42.0,
            ),
        ],
        recommended_actions=[
            "Investigate CPU-side preprocessing.",
            "Profile the data-loading pipeline.",
        ],
    )

    result = AgentRunResult(
        answer="Investigation complete.",
        state=AgentState(),
        findings=[finding],
    )

    formatter = InvestigationResultFormatter()

    output = formatter.format(result)

    assert "Investigation Summary" in output
    assert "Root Cause: CPU-side preprocessing bottleneck" in output
    assert "Confidence: 91%" in output
    assert "Explanation:" in output
    assert (
        "CPU utilization is high while GPU utilization remains low."
        in output
    )
    assert "Supporting Evidence:" in output
    assert "1. cpu_utilization is 96.0" in output
    assert "2. gpu_utilization is 42.0" in output
    assert "Recommended Actions:" in output
    assert "1. Investigate CPU-side preprocessing." in output
    assert "2. Profile the data-loading pipeline." in output


def test_formats_finding_without_evidence():
    finding = InvestigationFinding(
        hypothesis="Unknown bottleneck",
        confidence=0.45,
        explanation="The available telemetry is inconclusive.",
    )

    result = AgentRunResult(
        answer="Investigation complete.",
        state=AgentState(),
        findings=[finding],
    )

    formatter = InvestigationResultFormatter()

    output = formatter.format(result)

    assert "Root Cause: Unknown bottleneck" in output
    assert "Confidence: 45%" in output
    assert "The available telemetry is inconclusive." in output
    assert "Supporting Evidence:" not in output


def test_formats_finding_without_recommended_actions():
    finding = InvestigationFinding(
        hypothesis="GPU memory pressure",
        confidence=0.80,
        explanation="GPU memory utilization is elevated.",
        evidence=[
            InvestigationEvidence(
                source="get_service_health",
                description="gpu_memory_utilization is 95.0",
                value=95.0,
            )
        ],
    )

    result = AgentRunResult(
        answer="Investigation complete.",
        state=AgentState(),
        findings=[finding],
    )

    formatter = InvestigationResultFormatter()

    output = formatter.format(result)

    assert "Root Cause: GPU memory pressure" in output
    assert "Confidence: 80%" in output
    assert "Supporting Evidence:" in output
    assert "1. gpu_memory_utilization is 95.0" in output
    assert "Recommended Actions:" not in output


def test_returns_original_answer_when_no_findings():
    result = AgentRunResult(
        answer="I could not determine the root cause.",
        state=AgentState(),
        findings=[],
    )

    formatter = InvestigationResultFormatter()

    output = formatter.format(result)

    assert output == "I could not determine the root cause."


def test_formats_confidence_as_percentage():
    finding = InvestigationFinding(
        hypothesis="Network bottleneck",
        confidence=0.876,
        explanation="Network throughput is degraded.",
    )

    result = AgentRunResult(
        answer="Investigation complete.",
        state=AgentState(),
        findings=[finding],
    )

    formatter = InvestigationResultFormatter()

    output = formatter.format(result)

    assert "Confidence: 88%" in output