from investigation.analyzer import IncidentAnalyzer
from telemetry.scenarios import load_cpu_bottleneck_scenario


def test_investigation_evidence_is_chronological():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    timestamps = [
        item.timestamp
        for item in report.evidence
    ]

    assert timestamps == sorted(timestamps)

def test_investigation_generates_timeline():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert len(report.timeline) > 0
    assert len(report.timeline) == len(report.evidence)

def test_timeline_is_chronological():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    timestamps = [
        event.timestamp
        for event in report.timeline
    ]

    assert timestamps == sorted(timestamps)

def test_cpu_bottleneck_root_cause_is_preserved():
    incident, repository = load_cpu_bottleneck_scenario()

    analyzer = IncidentAnalyzer(repository)

    report = analyzer.investigate(incident)

    assert report.likely_root_cause == (
        "CPU-side preprocessing bottleneck"
    )

    assert report.confidence == 0.87