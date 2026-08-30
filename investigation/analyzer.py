from telemetry.models import Incident
from telemetry.repository import TelemetryRepository

from .models import Evidence, Hypothesis, InvestigationReport

from investigation.hypotheses.base import HypothesisDetector
from investigation.hypotheses.cpu_bottleneck import CPUBottleneckDetector
from investigation.hypotheses.gpu_oom import GPUOOMDetector
from investigation.hypotheses.network_bottleneck import (
    NetworkBottleneckDetector,
)


class IncidentAnalyzer:
    """Deterministic analyzer for ML service incidents."""

    def __init__(
        self,
        repository: TelemetryRepository,
        detectors: list[HypothesisDetector] | None = None,
    ) -> None:
        self.repository = repository

        self.detectors = detectors or [
            CPUBottleneckDetector(),
            GPUOOMDetector(),
            NetworkBottleneckDetector(),
        ]

    def investigate(self, incident: Incident) -> InvestigationReport:
        """Investigate an incident using registered hypothesis detectors."""

        hypotheses: list[Hypothesis] = []

        for detector in self.detectors:
            hypothesis = detector.detect(
                incident,
                self.repository,
            )

            if hypothesis is not None:
                hypotheses.append(hypothesis)

        # Highest-confidence hypothesis becomes the likely root cause.
        hypotheses.sort(
            key=lambda hypothesis: hypothesis.confidence,
            reverse=True,
        )

        if hypotheses:
            root_hypothesis = hypotheses[0]
            root_cause = root_hypothesis.name
            confidence = root_hypothesis.confidence

            evidence: list[Evidence] = []

            for hypothesis in hypotheses:
                evidence.extend(hypothesis.evidence)

            recommended_actions = root_hypothesis.recommended_actions

        else:
            root_cause = "Unable to determine root cause"
            confidence = 0.0
            evidence = []
            recommended_actions = []

        return InvestigationReport(
            incident_id=incident.incident_id,
            service_name=incident.service_name,
            started_at=incident.started_at,
            likely_root_cause=root_cause,
            confidence=confidence,
            evidence=evidence,
            hypotheses=hypotheses,
            recommended_actions=recommended_actions,
        )

