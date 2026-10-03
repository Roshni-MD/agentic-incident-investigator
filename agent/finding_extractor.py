from abc import ABC, abstractmethod

from .models import InvestigationEvidence, InvestigationFinding


class FindingExtractor(ABC):
    """Extract structured findings from collected investigation evidence."""

    @abstractmethod
    def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        """Return findings derived from investigation evidence."""
        raise NotImplementedError


class EvidenceFindingExtractor(FindingExtractor):
    """Create an initial structured finding from collected evidence."""

    def extract(
        self,
        evidence: list[InvestigationEvidence],
    ) -> list[InvestigationFinding]:
        if not evidence:
            return []

        return [
            InvestigationFinding(
                hypothesis="Evidence requires further analysis",
                confidence=0.0,
                explanation=(
                    "The investigation collected telemetry evidence, "
                    "but no root-cause hypothesis has been established yet."
                ),
                evidence=evidence,
            )
        ]