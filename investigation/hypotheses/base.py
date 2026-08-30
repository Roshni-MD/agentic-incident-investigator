from abc import ABC, abstractmethod

from telemetry.models import Incident
from telemetry.repository import TelemetryRepository

from investigation.models import Hypothesis


class HypothesisDetector(ABC):
    """Base interface for incident hypothesis detectors."""

    @abstractmethod
    def detect(
        self,
        incident: Incident,
        repository: TelemetryRepository,
    ) -> Hypothesis | None:
        """Return a hypothesis if the detector finds supporting evidence."""
        raise NotImplementedError