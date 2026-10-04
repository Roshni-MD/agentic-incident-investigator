from typing import Any
from unittest import result

from .models import InvestigationEvidence


class EvidenceCollector:
    """Converts tool results into structured investigation evidence."""

    def collect(
        self,
        tool_name: str,
        result: Any,
    ) -> list[InvestigationEvidence]:
        if tool_name == "get_current_metric":
            return self._collect_current_metric(result)

        if tool_name == "get_service_health":
            return self._collect_service_health(result)

        if tool_name == "get_metric_history":
            return self._collect_metric_history(result)

        if tool_name == "query_logs":
            return self._collect_query_logs(result)

        if tool_name == "get_recent_deployments":
            return self._collect_recent_deployments(result)

        return [
            InvestigationEvidence(
                source=tool_name,
                description=f"Result returned by {tool_name}",
                value=result,
            )
        ]

    def _collect_current_metric(
        self,
        result: Any,
    ) -> list[InvestigationEvidence]:
        service_name = result.get("service_name")
        metric_name = result.get("metric_name") or result.get("metric")
        value = result.get("value")

        if metric_name is not None and service_name is not None:
            description = (
                f"{metric_name} for {service_name} is {value}"
            )
        elif metric_name is not None:
            description = f"{metric_name} is {value}"
        else:
            description = "Result returned by get_current_metric"

        return [
            InvestigationEvidence(
                source="get_current_metric",
                description=description,
                value=value,
            )
        ]

    def _collect_service_health(
        self,
        result: Any,
    ) -> list[InvestigationEvidence]:
        evidence = []

        # Production format:
        # {
        #     "service_name": "...",
        #     "status": "ok",
        #     "metrics": {
        #         "cpu_utilization": {
        #             "value": 96.0,
        #             "timestamp": "..."
        #         },
        #         ...
        #     }
        # }
        metrics = result.get("metrics") if isinstance(result, dict) else None

        if isinstance(metrics, dict):
            for metric_name, metric_data in metrics.items():
                value = metric_data.get("value")
                timestamp = metric_data.get("timestamp")

                evidence.append(
                    InvestigationEvidence(
                        source="get_service_health",
                        description=(
                            f"{metric_name} is {value}"
                            f" at {timestamp}"
                        ),
                        value=value,
                    )
                )

            return evidence

        # Flat format used by the test:
        # {
        #     "cpu": 96,
        #     "gpu": 42,
        #     ...
        # }
        if isinstance(result, dict):
            for metric_name, value in result.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    evidence.append(
                        InvestigationEvidence(
                            source="get_service_health",
                            description=f"{metric_name} is {value}",
                            value=value,
                        )
                    )

        if evidence:
            return evidence

        # Unknown/unexpected format
        return [
            InvestigationEvidence(
                source="get_service_health",
                description="Result returned by get_service_health",
                value=result,
            )
        ]


    def _collect_metric_history(
        self,
        result: Any,
    ) -> list[InvestigationEvidence]:
        evidence = []

        for point in result:
            timestamp = point.get("timestamp")
            value = point.get("value")

            evidence.append(
                InvestigationEvidence(
                    source="get_metric_history",
                    description=(
                        f"Metric value is {value} at {timestamp}"
                    ),
                    value=value,
                )
            )

        return evidence

    def _collect_query_logs(
        self,
        result: Any,
    ) -> list[InvestigationEvidence]:
        evidence = []

        for log in result:
            timestamp = log.get("timestamp")
            level = log.get("level")
            message = log.get("message")

            description = (
                f"{level} log at {timestamp}: {message}"
            )

            evidence.append(
                InvestigationEvidence(
                    source="query_logs",
                    description=description,
                    value=log,
                )
            )

        return evidence

    def _collect_recent_deployments(
        self,
        result: Any,
    ) -> list[InvestigationEvidence]:
        evidence = []

        for deployment in result:
            deployment_id = deployment.get("deployment_id")
            service_name = deployment.get("service_name")
            model_name = deployment.get("model_name")
            model_version = deployment.get("model_version")
            previous_version = deployment.get("previous_version")
            timestamp = deployment.get("timestamp")

            description = (
                f"Deployment {deployment_id} for {service_name}: "
                f"{model_name} changed from "
                f"{previous_version} to {model_version} "
                f"at {timestamp}"
            )

            evidence.append(
                InvestigationEvidence(
                    source="get_recent_deployments",
                    description=description,
                    value=deployment,
                )
            )

        return evidence

