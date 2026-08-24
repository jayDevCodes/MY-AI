from __future__ import annotations

from dataclasses import dataclass

from .agent_graph import ExecutionBudget
from .capability_ledger import CapabilityLedger
from .evolution import StrategyScore
from .model_router import RoutingDecision, RoutingRequest


@dataclass(frozen=True)
class CognitiveComputePolicy:
    reasoning_depth: int
    max_parallel: int
    max_retries: int
    verification_passes: int
    preferred_tier: str
    reason: str

    def execution_budget(self, *, base: ExecutionBudget | None = None) -> ExecutionBudget:
        base = base or ExecutionBudget()
        return ExecutionBudget(
            max_depth=max(1, min(base.max_depth, self.reasoning_depth)),
            max_nodes=max(1, base.max_nodes),
            max_parallel=max(1, min(base.max_parallel, self.max_parallel)),
            max_retries=max(0, min(base.max_retries, self.max_retries)),
        )


class CognitiveComputeController:
    """Allocate bounded test-time compute from measured capability gaps and strategy evidence."""

    def __init__(
        self,
        ledger: CapabilityLedger | None = None,
        strategy_scores: tuple[StrategyScore, ...] = (),
    ) -> None:
        self.ledger = ledger
        self.strategy_scores = strategy_scores

    def policy_for(
        self,
        request: RoutingRequest,
        decision: RoutingDecision,
    ) -> CognitiveComputePolicy:
        gap = self._capability_gap(request.task_kind)
        strategy_reliability = self._strategy_reliability(request.task_kind)
        pressure = max(gap, request.uncertainty, 0.0)
        if request.risk == "high":
            pressure = max(pressure, 0.85)

        depth = 1 + round(2 * pressure)
        retries = 0 if pressure < 0.4 else 1
        verification = 1 if pressure < 0.65 else 2
        parallel = 1 if request.latency_sensitive and pressure < 0.6 else (2 if pressure < 0.8 else 3)

        if strategy_reliability >= 0.9 and pressure < 0.7:
            parallel = max(1, parallel - 1)
        if strategy_reliability < 0.6:
            depth = min(3, depth + 1)
            verification = min(2, verification + 1)

        return CognitiveComputePolicy(
            reasoning_depth=max(1, min(3, depth)),
            max_parallel=max(1, min(3, parallel)),
            max_retries=max(0, min(1, retries)),
            verification_passes=max(1, min(2, verification)),
            preferred_tier=decision.tier,
            reason=(
                f"tier={decision.tier}; capability_gap={gap:.2f}; "
                f"strategy_reliability={strategy_reliability:.2f}; pressure={pressure:.2f}"
            ),
        )

    def _capability_gap(self, task_kind: str) -> float:
        dimension = {
            "research": "reasoning",
            "reasoning": "reasoning",
            "architecture": "planning",
            "coding": "coding",
            "code_review": "verification",
            "debugging": "debugging",
            "memory": "memory",
        }.get(task_kind)
        if not dimension or self.ledger is None or self.ledger.current is None:
            return 0.0
        score = self.ledger.current.score(dimension)
        if score is None:
            return 0.0
        return max(0.0, min(1.0, 1.0 - score))

    def _strategy_reliability(self, task_kind: str) -> float:
        if not self.strategy_scores:
            return 0.0
        relevant = [item for item in self.strategy_scores if task_kind.casefold() in item.strategy.casefold()]
        items = relevant or list(self.strategy_scores)
        return max(0.0, min(1.0, max(item.success_rate for item in items)))
