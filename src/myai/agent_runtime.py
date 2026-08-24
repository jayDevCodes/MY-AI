from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .agent_graph import ExecutionBudget, JudgeVerdict, RecursiveAgentGraph, TaskNode, WorkArtifact
from .model_router import AdaptiveModelRouter, RoutingRequest
from .provider_pool import ModelTier, TieredModelPool
from .schemas import ChatMessage


@dataclass(frozen=True)
class AgentRuntimeResult:
    artifact: WorkArtifact
    tier_by_task: tuple[tuple[str, ModelTier], ...]


class MultiModelAgentRuntime:
    """Connect the recursive graph to real tier-specific model endpoints."""

    def __init__(self, pool: TieredModelPool, router: AdaptiveModelRouter, budget: ExecutionBudget) -> None:
        self.pool = pool
        self.router = router
        self.budget = budget
        self._tiers: dict[str, ModelTier] = {}

    def run(self, objective: str, task_kind: str, context: Sequence[ChatMessage] = ()) -> AgentRuntimeResult:
        self._tiers.clear()
        graph = RecursiveAgentGraph(self.budget)
        root = TaskNode("root", objective, role=task_kind)
        result = graph.run(
            root,
            decompose=lambda node, budget: self._decompose(node, budget),
            worker=lambda node, children: self._worker(node, children, context),
            judge=lambda node, artifact, children: self._judge(node, artifact, children),
        )
        return AgentRuntimeResult(result, tuple(sorted(self._tiers.items())))

    def _decompose(self, node: TaskNode, budget: ExecutionBudget) -> Sequence[TaskNode]:
        if node.depth > 0:
            return ()
        roles = {
            "research": ("evidence", "countercheck", "synthesis"),
            "reasoning": ("solver_a", "solver_b", "critic"),
            "coding": ("code_mapper", "implementer", "reviewer"),
        }.get(node.role, ("analyst", "critic"))
        return tuple(
            TaskNode(
                id=f"{node.id}:{role}",
                objective=f"{role}: {node.objective}",
                role=role,
                depth=node.depth + 1,
                parent_id=node.id,
            )
            for role in roles[: max(1, min(len(roles), budget.max_parallel))]
        )

    def _choose_tier(self, node: TaskNode, context: Sequence[ChatMessage]) -> ModelTier:
        decision = self.router.choose(
            RoutingRequest(
                task_kind=node.role,
                complexity=0.85 if node.depth == 0 else 0.7,
                uncertainty=0.75 if node.role in {"critic", "reviewer", "countercheck"} else 0.45,
                context_size=sum(len(m.content) for m in context),
                risk="high" if node.role in {"reviewer", "countercheck", "synthesis"} else "medium",
                latency_sensitive=False,
                quality_priority=node.role in {"critic", "reviewer", "synthesis"},
            )
        )
        self._tiers[node.id] = decision.tier
        return decision.tier

    def _worker(
        self,
        node: TaskNode,
        children: Sequence[WorkArtifact],
        context: Sequence[ChatMessage],
    ) -> WorkArtifact:
        tier = self._choose_tier(node, context)
        child_text = "\n".join(
            f"[{child.task_id} confidence={child.confidence:.2f}] {child.output}"
            for child in children
        )
        system = ChatMessage(
            role="system",
            content=(
                "You are a specialist worker in MY-AI V7.1. Return only useful work for the "
                "assigned role. Do not claim tools or evidence you did not receive. Preserve "
                "uncertainty and disagreements."
            ),
        )
        prompt = ChatMessage(
            role="user",
            content=(
                f"ROLE: {node.role}\nOBJECTIVE: {node.objective}\n"
                f"PRIOR SPECIALIST ARTIFACTS:\n{child_text or '(none)'}"
            ),
        )
        output = self.pool.generate(tier, [system, *context, prompt])
        return WorkArtifact(
            task_id=node.id,
            role=node.role,
            output=output,
            confidence=0.6,
            findings=(f"executed_on:{tier}",),
        )

    def _judge(
        self,
        node: TaskNode,
        artifact: WorkArtifact,
        children: Sequence[WorkArtifact],
    ) -> JudgeVerdict:
        tier: ModelTier = "frontier"
        self._tiers[f"{node.id}:judge"] = tier
        child_summary = "\n".join(
            f"[{child.task_id}] {child.output}" for child in children
        )
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are an independent judge. Evaluate whether the worker output is "
                    "consistent, useful, evidence-aware and complete. Return JSON only: "
                    '{"passed":true|false,"confidence":0.0-1.0,"feedback":"..."}'
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"TASK: {node.objective}\nWORKER:\n{artifact.output}\n"
                    f"CHILDREN:\n{child_summary or '(none)'}"
                ),
            ),
        ]
        raw = self.pool.generate(tier, messages)
        try:
            data = json.loads(raw)
            return JudgeVerdict(
                bool(data.get("passed")),
                max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                str(data.get("feedback", "")),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            passed = bool(raw.strip()) and "error" not in raw.casefold()
            return JudgeVerdict(passed, 0.5 if passed else 0.0, "Judge output was not valid JSON.")
