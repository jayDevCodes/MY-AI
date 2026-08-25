from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from .agent_graph import ExecutionBudget, JudgeVerdict, RecursiveAgentGraph, TaskNode, WorkArtifact
from .cognitive_state import CognitiveState
from .model_router import AdaptiveModelRouter, RoutingRequest
from .provider_pool import ModelTier, TieredModelPool
from .schemas import ChatMessage


@dataclass(frozen=True)
class AgentRuntimeResult:
    artifact: WorkArtifact
    tier_by_task: tuple[tuple[str, ModelTier], ...]


class MultiModelAgentRuntime:
    """Connect the recursive graph to real tier-specific model endpoints."""

    def __init__(
        self,
        pool: TieredModelPool,
        router: AdaptiveModelRouter,
        budget: ExecutionBudget,
        worker_tier_override: ModelTier | None = None,
    ) -> None:
        self.pool = pool
        self.router = router
        self.budget = budget
        self.worker_tier_override = worker_tier_override
        self._tiers: dict[str, ModelTier] = {}

    def run(
        self,
        objective: str,
        task_kind: str,
        context: Sequence[ChatMessage] = (),
        state: CognitiveState | None = None,
    ) -> AgentRuntimeResult:
        self._tiers.clear()
        graph = RecursiveAgentGraph(self.budget)
        root = TaskNode("root", objective, role=task_kind)
        result = graph.run(
            root,
            decompose=lambda node, budget: self._decompose(node, budget),
            worker=lambda node, children: self._worker(node, children, context, state),
            judge=lambda node, artifact, children: self._judge(node, artifact, children, state),
        )
        if state is not None:
            state.active_strategy = f"recursive-agent:{task_kind}"
            state.observe(f"agent-runtime-complete:{task_kind}:{result.confidence:.2f}")
            state.set_uncertainty(max(0.0, min(1.0, 1.0 - result.confidence)))
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
        if self.worker_tier_override is not None and node.role not in {"critic", "reviewer", "countercheck", "synthesis"}:
            tier = self.worker_tier_override
        else:
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
            tier = decision.tier
        self._tiers[node.id] = tier
        return tier

    def _worker(
        self,
        node: TaskNode,
        children: Sequence[WorkArtifact],
        context: Sequence[ChatMessage],
        state: CognitiveState | None = None,
    ) -> WorkArtifact:
        tier = self._choose_tier(node, context)
        child_text = "\n".join(
            f"[{child.task_id} confidence={child.confidence:.2f}] {child.output}"
            for child in children
        )
        state_text = state.summary() if state is not None else "(no shared cognitive state)"
        system = ChatMessage(
            role="system",
            content=(
                "You are a specialist worker in MY-AI V10. Return only useful work for the assigned role. "
                "Do not claim tools or evidence you did not receive. Preserve uncertainty and disagreements. "
                "Use the shared cognitive state as context, not as proof; distinguish beliefs from observations and avoid inventing missing facts."
            ),
        )
        prompt = ChatMessage(
            role="user",
            content=(
                f"ROLE: {node.role}\nOBJECTIVE: {node.objective}\n"
                f"COGNITIVE STATE:\n{state_text}\n"
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
        state: CognitiveState | None = None,
    ) -> JudgeVerdict:
        tier: ModelTier = "frontier"
        self._tiers[f"{node.id}:judge"] = tier
        child_summary = "\n".join(
            f"[{child.task_id}] {child.output}" for child in children
        )
        state_text = state.summary() if state is not None else "(no shared cognitive state)"
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are an independent judge for MY-AI V10. Evaluate whether the worker output is "
                    "consistent, useful, evidence-aware and complete. Treat the cognitive state as "
                    "context rather than ground truth. Return JSON only: "
                    '{"passed":true|false,"confidence":0.0-1.0,"feedback":"..."}'
                ),
            ),
            ChatMessage(
                role="user",
                content=(
                    f"TASK: {node.objective}\nCOGNITIVE STATE:\n{state_text}\n"
                    f"WORKER:\n{artifact.output}\nCHILDREN:\n{child_summary or '(none)'}"
                ),
            ),
        ]
        raw = self.pool.generate(tier, messages)
        try:
            data = json.loads(raw)
            verdict = JudgeVerdict(
                bool(data.get("passed")),
                max(0.0, min(1.0, float(data.get("confidence", 0.0)))),
                str(data.get("feedback", "")),
            )
        except (TypeError, ValueError, json.JSONDecodeError):
            passed = bool(raw.strip()) and "error" not in raw.casefold()
            verdict = JudgeVerdict(passed, 0.5 if passed else 0.0, "Judge output was not valid JSON.")
        if state is not None:
            state.observe(f"judge:{node.role}:{'passed' if verdict.passed else 'failed'}:{verdict.confidence:.2f}")
        return verdict
