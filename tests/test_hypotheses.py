from telemetry.scenarios import (
    load_cpu_bottleneck_scenario,
    load_gpu_oom_scenario,
)

from investigation.hypotheses.cpu_bottleneck import (
    CPUBottleneckDetector,
)
from investigation.hypotheses.gpu_oom import GPUOOMDetector


def test_cpu_bottleneck_detector_detects_incident():
    incident, repository = load_cpu_bottleneck_scenario()

    detector = CPUBottleneckDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None
    assert hypothesis.name == "CPU-side preprocessing bottleneck"
    assert hypothesis.confidence == 0.87


def test_cpu_bottleneck_detector_returns_evidence():
    incident, repository = load_cpu_bottleneck_scenario()

    detector = CPUBottleneckDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None

    assert len(hypothesis.evidence) == 3

    sources = {
        evidence.source
        for evidence in hypothesis.evidence
    }

    assert sources == {"metrics"}

def test_gpu_oom_detector():
    incident, repository = load_gpu_oom_scenario()

    detector = GPUOOMDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None
    assert hypothesis.name == "GPU out-of-memory"
    assert hypothesis.confidence == 0.98


def test_gpu_oom_detector_has_evidence():
    incident, repository = load_gpu_oom_scenario()

    detector = GPUOOMDetector()

    hypothesis = detector.detect(
        incident,
        repository,
    )

    assert hypothesis is not None

    assert len(hypothesis.evidence) == 3

    sources = {
        evidence.source
        for evidence in hypothesis.evidence
    }

    assert sources == {
        "metrics",
        "logs",
        "deployment",
    }