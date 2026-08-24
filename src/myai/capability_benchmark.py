from __future__ import annotations

from dataclasses import dataclass

from .capability_ledger import DIMENSIONS, CapabilityScore, CapabilitySnapshot, now_iso


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    dimension: str
    description: str
    success_metric: str


DEFAULT_CASES = (
    BenchmarkCase(
        "recursive-agent-graph",
        "planning",
        "Decompose and execute bounded specialist work.",
        "artifact success",
    ),
    BenchmarkCase(
        "frontier-routing",
        "reasoning",
        "Escalate high-uncertainty work to the frontier tier.",
        "correct tier",
    ),
    BenchmarkCase(
        "program-graph-localization",
        "repository_understanding",
        "Localize a target through program/data/control context.",
        "target in slice",
    ),
    BenchmarkCase(
        "causal-trace-diagnosis",
        "debugging",
        "Map runtime failure to a compact causal impact slice.",
        "frame + impact evidence",
    ),
    BenchmarkCase(
        "cross-episode-repair-memory",
        "memory",
        "Reuse a validated repair strategy across episodes.",
        "successful retrieval",
    ),
    BenchmarkCase(
        "trace-verification",
        "verification",
        "Reject or accept a repair using execution evidence.",
        "validated outcome",
    ),
    BenchmarkCase(
        "targeted-context",
        "efficiency",
        "Avoid unrelated repository context for a targeted task.",
        "context reduction",
    ),
    BenchmarkCase(
        "strategy-promotion",
        "self_improvement",
        "Promote only measured improvements over baseline.",
        "promotion gate",
    ),
)


class CapabilityBenchmark:
    """Registry and snapshot builder; scores must come from executed evidence."""

    def __init__(self, cases: tuple[BenchmarkCase, ...] = DEFAULT_CASES) -> None:
        self.cases = cases

    def snapshot(
        self,
        *,
        version: str,
        commit: str,
        results: dict[str, tuple[float, tuple[str, ...]]],
        notes: tuple[str, ...] = (),
    ) -> CapabilitySnapshot:
        scores: list[CapabilityScore] = []
        for dimension in DIMENSIONS:
            matching = [
                case
                for case in self.cases
                if case.dimension == dimension and case.name in results
            ]
            values = [results[case.name] for case in matching]
            if values:
                value = sum(item[0] for item in values) / len(values)
                evidence = tuple(
                    evidence_item
                    for _, evidence_items in values
                    for evidence_item in evidence_items
                )
                status = "measured"
            else:
                value = 0.0
                evidence = ()
                status = "unmeasured"
            scores.append(
                CapabilityScore(
                    dimension,
                    round(value, 4),
                    status,
                    evidence=evidence,
                )
            )

        passes = sum(1 for score, _ in results.values() if score >= 0.8)
        return CapabilitySnapshot(
            version=version,
            commit=commit,
            captured_at=now_iso(),
            scores=tuple(scores),
            benchmark_count=len(results),
            benchmark_passes=passes,
            regression_count=0,
            notes=notes,
        )
