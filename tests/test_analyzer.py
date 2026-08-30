from investigation.analyzer import IncidentAnalyzer
from telemetry.scenarios import (
    load_cpu_bottleneck_scenario,
    load_gpu_oom_scenario,
)


def test_cpu_bottleneck_investigation():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert report.likely_root_cause == "CPU-side preprocessing bottleneck"
    assert report.confidence == 0.87

    assert len(report.hypotheses) >= 1

    assert report.recommended_actions


def test_gpu_oom_investigation():
    incident, repository = load_gpu_oom_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert report.likely_root_cause == "GPU out-of-memory"
    assert report.confidence == 0.98

    assert len(report.hypotheses) >= 1

    assert report.recommended_actions


def test_hypotheses_are_ranked_by_confidence():
    incident, repository = load_gpu_oom_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    confidences = [
        hypothesis.confidence
        for hypothesis in report.hypotheses
    ]

    assert confidences == sorted(
        confidences,
        reverse=True,
    )


def test_cpu_investigation_has_cpu_actions():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert any(
        "CPU" in action
        for action in report.recommended_actions
    )


def test_gpu_oom_investigation_has_gpu_actions():
    incident, repository = load_gpu_oom_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert any(
        "GPU" in action
        for action in report.recommended_actions
    )