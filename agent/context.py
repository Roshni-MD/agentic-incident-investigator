from telemetry.models import Incident

from .models import InvestigationContext


def build_investigation_context(
    incident: Incident,
) -> InvestigationContext:
    return InvestigationContext(
        incident_id=incident.incident_id,
        service_name=incident.service_name,
        incident_type=incident.incident_type.value,
        started_at=incident.started_at.isoformat(),
        available_tools=[
            "get_current_metric",
            "get_metric_history",
            "get_service_health",
            "query_logs",
            "get_recent_deployments",
        ],
    )