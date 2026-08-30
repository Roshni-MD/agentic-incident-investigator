from investigation.hypotheses.base import HypothesisDetector
from investigation.models import Evidence, Hypothesis

from telemetry.models import Incident
from telemetry.repository import TelemetryRepository


class CPUBottleneckDetector(HypothesisDetector):

    def detect(
        self,
        incident: Incident,
        repository: TelemetryRepository,
    ) -> Hypothesis | None:

        metrics = repository.get_metrics(
            service_name=incident.service_name,
            start_time=incident.metrics[0].timestamp,
            end_time=incident.metrics[-1].timestamp,
        )

        if len(metrics) < 2:
            return None

        first = metrics[0]
        last = metrics[-1]

        if not (
            first.cpu_utilization < 80
            and last.cpu_utilization >= 90
            and first.gpu_utilization > 80
            and last.gpu_utilization < 60
            and last.data_loading_ms > first.data_loading_ms * 5
        ):
            return None

        evidence = [
            Evidence(
                source="metrics",
                description=(
                    f"CPU utilization increased from "
                    f"{first.cpu_utilization}% to "
                    f"{last.cpu_utilization}%."
                ),
                timestamp=last.timestamp,
            ),
            Evidence(
                source="metrics",
                description=(
                    f"GPU utilization decreased from "
                    f"{first.gpu_utilization}% to "
                    f"{last.gpu_utilization}%."
                ),
                timestamp=last.timestamp,
            ),
            Evidence(
                source="metrics",
                description=(
                    f"Data loading latency increased from "
                    f"{first.data_loading_ms} ms to "
                    f"{last.data_loading_ms} ms."
                ),
                timestamp=last.timestamp,
            ),
        ]

        return Hypothesis(
            name="CPU-side preprocessing bottleneck",
            confidence=0.87,
            explanation=(
                "CPU saturation and increased data-loading latency "
                "coincide with reduced GPU utilization."
            ),
            evidence=evidence,
            recommended_actions=[
                "Investigate CPU-side preprocessing latency.",
                "Compare preprocessing implementation between deployed versions.",
                "Consider rolling back the preprocessing component if confirmed.",
            ],
        )