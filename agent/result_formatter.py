from .state import AgentRunResult


class InvestigationResultFormatter:
    def format(self, result: AgentRunResult) -> str:
        """
        Investigation Summary
        ---------------------
        Root Cause:
        CPU-side preprocessing bottleneck

        Confidence:
        91%

        Why:
        CPU utilization reached 96% while GPU utilization remained at 42%.
        Data loading latency was also elevated.

        Supporting Evidence:
        1. cpu_utilization = 96.0
        2. gpu_utilization = 42.0
        3. data_loading_ms = 380.0

        Recommended Actions:
        1. Investigate CPU-side preprocessing.
        """

        if not result.findings:
            return result.answer

        finding = result.findings[0]

        lines = [
            "Investigation Summary",
            "---------------------",
            f"Root Cause: {finding.hypothesis}",
            f"Confidence: {finding.confidence:.0%}",
            "",
            "Explanation:",
            finding.explanation,
        ]

        if finding.evidence:
            lines.extend([
                "",
                "Supporting Evidence:",
            ])

            for index, evidence in enumerate(finding.evidence, start=1):
                lines.append(
                    f"{index}. {evidence.description}"
                )

        if finding.recommended_actions:
            lines.extend([
                "",
                "Recommended Actions:",
            ])

            for index, action in enumerate(
                finding.recommended_actions,
                start=1,
            ):
                lines.append(f"{index}. {action}")

        return "\n".join(lines)