from investigation.analyzer import IncidentAnalyzer
from investigation.hypotheses.network_bottleneck import (
    NetworkBottleneckDetector,
)
from telemetry.scenarios import load_network_bottleneck_scenario


def test_network_bottleneck_detector():
    incident, repository = load_network_bottleneck_scenario()

    detector = NetworkBottleneckDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None
    assert hypothesis.name == "Network bottleneck"
    assert hypothesis.confidence == 0.96


def test_network_bottleneck_has_metric_evidence():
    incident, repository = load_network_bottleneck_scenario()

    detector = NetworkBottleneckDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None

    sources = {
        evidence.source
        for evidence in hypothesis.evidence
    }

    assert "metrics" in sources


def test_network_bottleneck_investigation():
    incident, repository = load_network_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert report.likely_root_cause == "Network bottleneck"
    assert report.confidence == 0.96

    assert report.recommended_actions

    assert any(
        "network" in action.lower()
        for action in report.recommended_actions
    )