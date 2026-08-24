from dataclasses import dataclass
from typing import Literal


ModelTier = Literal["fast", "balanced", "frontier"]
TaskRisk = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class RoutingRequest:
    task_kind: str
    complexity: float = 0.5
    uncertainty: float = 0.0
    context_size: int = 0
    risk: TaskRisk = "low"
    latency_sensitive: bool = False
    quality_priority: bool = True


@dataclass(frozen=True)
class RoutingDecision:
    tier: ModelTier
    reason: str
    cache_context: bool
    allow_parallel: bool


class AdaptiveModelRouter:
    """Budget-aware routing instead of permanently binding roles to one model."""

    def choose(self, request: RoutingRequest) -> RoutingDecision:
        score = max(
            0.0,
            min(1.0, 0.55 * request.complexity + 0.45 * request.uncertainty),
        )
        if request.risk == "high":
            score = max(score, 0.9)
        if request.task_kind in {"research", "reasoning", "architecture", "code_review"}:
            score = max(score, 0.75)
        if request.task_kind in {"formatting", "classification", "summarization"}:
            score = min(score, 0.4)

        if request.latency_sensitive and score < 0.65:
            tier: ModelTier = "fast"
        elif score >= 0.78:
            tier = "frontier"
        elif score >= 0.42:
            tier = "balanced"
        else:
            tier = "fast"

        return RoutingDecision(
            tier=tier,
            reason=(
                f"score={score:.2f}; complexity={request.complexity:.2f}; "
                f"uncertainty={request.uncertainty:.2f}; risk={request.risk}"
            ),
            cache_context=request.context_size >= 1500,
            allow_parallel=request.complexity >= 0.55 and request.task_kind not in {"chat"},
        )
