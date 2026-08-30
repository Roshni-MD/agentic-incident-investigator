from investigation.hypotheses.base import HypothesisDetector
from investigation.models import Evidence, Hypothesis

from telemetry.models import Incident
from telemetry.repository import TelemetryRepository


class GPUOOMDetector(HypothesisDetector):
    """Detect GPU out-of-memory conditions."""

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

        evidence: list[Evidence] = []

        latest_metric = metrics[-1]

        # Strong signal: GPU memory is essentially saturated.
        gpu_memory_saturated = (
            latest_metric.gpu_memory_utilization >= 95
        )

        # Look for explicit GPU OOM messages in logs.
        oom_logs = [
            log
            for log in logs
            if "out of memory" in log.message.lower()
            or "cuda out of memory" in log.message.lower()
            or "oom" in log.message.lower()
        ]

        explicit_oom_signal = bool(oom_logs)

        # A deployment can provide useful supporting evidence because
        # increased model memory usage may have been introduced by a
        # new model version.
        recent_deployment = deployments[-1] if deployments else None

        if not gpu_memory_saturated and not explicit_oom_signal:
            return None

        if gpu_memory_saturated:
            evidence.append(
                Evidence(
                    source="metrics",
                    description=(
                        f"GPU memory utilization reached "
                        f"{latest_metric.gpu_memory_utilization}%."
                    ),
                    timestamp=latest_metric.timestamp,
                )
            )

        for log in oom_logs:
            evidence.append(
                Evidence(
                    source="logs",
                    description=f"{log.level}: {log.message}",
                    timestamp=log.timestamp,
                )
            )

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

        # Explicit OOM logs are stronger evidence than memory saturation
        # alone. Combining both signals gives very high confidence.
        if explicit_oom_signal and gpu_memory_saturated:
            confidence = 0.98
            explanation = (
                "GPU memory is critically saturated and logs contain "
                "explicit out-of-memory errors."
            )
        elif explicit_oom_signal:
            confidence = 0.95
            explanation = (
                "Logs contain explicit GPU out-of-memory errors."
            )
        else:
            confidence = 0.82
            explanation = (
                "GPU memory utilization is critically high, "
                "indicating likely GPU memory exhaustion."
            )

        return Hypothesis(
            name="GPU out-of-memory",
            confidence=confidence,
            explanation=explanation,
            evidence=evidence,
            recommended_actions=[
                "Inspect GPU memory usage and allocation patterns.",
                "Compare model memory requirements between deployed versions.",
                "Check whether batch size or input dimensions increased.",
                "Consider rolling back the deployment if the new version introduced the memory increase.",
            ],
        )

