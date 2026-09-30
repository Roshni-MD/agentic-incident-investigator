from telemetry.models import Incident

from .models import InvestigationContext
from .tools import AgentToolRegistry


def build_investigation_context(
    incident: Incident,
    tools: AgentToolRegistry,
) -> InvestigationContext:
    """Build structured context for an incident investigation."""
    return InvestigationContext(
        incident_id=incident.incident_id,
        service_name=incident.service_name,
        incident_type=incident.incident_type.value,
        started_at=incident.started_at.isoformat(),
        available_tools=tools.names(),
    )


def build_system_prompt(context: InvestigationContext) -> str:
    """Build the system prompt from investigation context."""
    return (
        "You are an ML infrastructure incident investigation agent. "
        "Investigate incidents using the available telemetry tools. "
        "Gather evidence before determining the root cause.\n\n"
        f"Investigation context:\n"
        f"{context.model_dump_json(indent=2)}"
    )