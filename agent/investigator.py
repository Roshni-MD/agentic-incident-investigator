from telemetry.models import Incident

from investigation.analyzer import IncidentAnalyzer
from investigation.models import InvestigationReport


class IncidentInvestigator:
    """
    Agentic interface for investigating ML service incidents.

    The initial implementation uses the deterministic analyzer.
    A future implementation can replace this reasoning layer with
    an LLM while keeping the same interface.
    """

    def __init__(self, analyzer: IncidentAnalyzer) -> None:
        self.analyzer = analyzer

    def investigate(
        self,
        incident: Incident,
    ) -> InvestigationReport:
        return self.analyzer.investigate(incident)
