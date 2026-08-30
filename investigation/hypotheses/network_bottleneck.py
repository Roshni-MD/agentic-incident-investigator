from investigation.hypotheses.base import HypothesisDetector
from investigation.models import Evidence, Hypothesis

from telemetry.models import Incident
from telemetry.repository import TelemetryRepository


class NetworkBottleneckDetector(HypothesisDetector):
    """Detect network bottleneck conditions."""

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

        logs = repository.get_logs(
            service_name=incident.service_name,
            start_time=incident.metrics[0].timestamp,
            end_time=incident.metrics[-1].timestamp,
        )

        deployments = repository.get_deployments(
            service_name=incident.service_name,
            start_time=incident.metrics[0].timestamp,
            end_time=incident.metrics[-1].timestamp,
        )

        if not metrics:
            return None

        first = metrics[0]
        latest = metrics[-1]

        evidence: list[Evidence] = []

        network_saturated = (
            latest.network_utilization >= 90
        )

        network_latency_high = (
            latest.network_latency_ms >= 100
        )

        network_logs = [
            log
            for log in logs
            if (
                "network" in log.message.lower()
                and (
                    "latency" in log.message.lower()
                    or "saturation" in log.message.lower()
                    or "congestion" in log.message.lower()
                )
            )
        ]

        explicit_network_signal = bool(network_logs)

        if not (
            network_saturated
            or network_latency_high
            or explicit_network_signal
        ):
            return None

        if network_saturated:
            evidence.append(
                Evidence(
                    source="metrics",
                    description=(
                        f"Network utilization reached "
                        f"{latest.network_utilization}%."
                    ),
                    timestamp=latest.timestamp,
                )
            )

        if network_latency_high:
            evidence.append(
                Evidence(
                    source="metrics",
                    description=(
                        f"Network latency increased to "
                        f"{latest.network_latency_ms} ms."
                    ),
                    timestamp=latest.timestamp,
                )
            )

        for log in network_logs:
            evidence.append(
                Evidence(
                    source="logs",
                    description=f"{log.level}: {log.message}",
                    timestamp=log.timestamp,
                )
            )

        recent_deployment = deployments[-1] if deployments else None

        if recent_deployment is not None:
            evidence.append(
                Evidence(
                    source="deployment",
                    description=(
                        f"Recent deployment "
                        f"{recent_deployment.deployment_id} changed "
                        f"{recent_deployment.model_name} from "
                        f"{recent_deployment.previous_version} to "
                        f"{recent_deployment.model_version}."
                    ),
                    timestamp=recent_deployment.timestamp,
                )
            )

        if network_saturated and network_latency_high:
            confidence = 0.96
            explanation = (
                "Network utilization is critically high and network "
                "latency has increased significantly."
            )
        elif explicit_network_signal and network_saturated:
            confidence = 0.95
            explanation = (
                "Network saturation is supported by explicit network "
                "warnings in the service logs."
            )
        elif network_latency_high:
            confidence = 0.85
            explanation = (
                "Elevated network latency indicates a possible "
                "network bottleneck."
            )
        else:
            confidence = 0.80
            explanation = (
                "Network utilization is elevated and may be limiting "
                "service throughput."
            )

        return Hypothesis(
            name="Network bottleneck",
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            recommended_actions=[
                "Inspect network bandwidth and interface utilization.",
                "Check network latency and packet loss between service components.",
                "Investigate recent changes to network configuration or traffic patterns.",
                "Consider shifting traffic or scaling network capacity if saturation is confirmed.",
            ],
        )