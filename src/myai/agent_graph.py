from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence


@dataclass(frozen=True)
class ExecutionBudget:
    max_depth: int = 3
    max_nodes: int = 32
    max_parallel: int = 4
    max_retries: int = 1


@dataclass(frozen=True)
class TaskNode:
    id: str
    objective: str
    role: str = "general"
    depth: int = 0
    parent_id: str | None = None


@dataclass(frozen=True)
class WorkArtifact:
    task_id: str
    role: str
    output: str
    confidence: float = 0.0
    findings: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    children: tuple["WorkArtifact", ...] = ()


@dataclass(frozen=True)
class JudgeVerdict:
    passed: bool
    confidence: float
    feedback: str = ""
    failed_task_ids: tuple[str, ...] = ()


class Decomposer(Protocol):
    def __call__(self, node: TaskNode, budget: ExecutionBudget) -> Sequence[TaskNode]: ...


class Worker(Protocol):
    def __call__(self, node: TaskNode, children: Sequence[WorkArtifact]) -> WorkArtifact: ...


class Judge(Protocol):
    def __call__(
        self,
        node: TaskNode,
        artifact: WorkArtifact,
        children: Sequence[WorkArtifact],
    ) -> JudgeVerdict: ...


class RecursiveAgentGraph:
    """Bounded hierarchical orchestration for heterogeneous model agents.

    The graph recursively decomposes a task, executes independent branches in
    parallel, synthesizes their structured artifacts, and retries only a failed
    node. Agents exchange compact artifacts rather than replaying full context.
    """

    def __init__(self, budget: ExecutionBudget | None = None) -> None:
        self.budget = budget or ExecutionBudget()
        self._visited: set[str] = set()
        self._node_count = 0

    def run(
        self,
        root: TaskNode,
        *,
        decompose: Decomposer,
        worker: Worker,
        judge: Judge,
    ) -> WorkArtifact:
        self._visited.clear()
        self._node_count = 0
        return self._solve(root, decompose=decompose, worker=worker, judge=judge)

    def _solve(
        self,
        node: TaskNode,
        *,
        decompose: Decomposer,
        worker: Worker,
        judge: Judge,
    ) -> WorkArtifact:
        if node.id in self._visited:
            return WorkArtifact(
                task_id=node.id,
                role=node.role,
                output="cycle_detected",
                confidence=0.0,
                open_questions=("task graph cycle detected",),
            )
        if node.depth > self.budget.max_depth or self._node_count >= self.budget.max_nodes:
            return WorkArtifact(
                task_id=node.id,
                role=node.role,
                output="budget_exhausted",
                confidence=0.0,
                open_questions=("execution budget exhausted",),
            )

        self._visited.add(node.id)
        self._node_count += 1
        children = tuple(decompose(node, self.budget))
        if children:
            child_results = self._solve_parallel(
                children, decompose=decompose, worker=worker, judge=judge
            )
        else:
            child_results = ()

        artifact = worker(node, child_results)
        verdict = judge(node, artifact, child_results)

        retries = 0
        while not verdict.passed and retries < self.budget.max_retries:
            retries += 1
            retry_node = TaskNode(
                id=node.id,
                objective=(
                    f"{node.objective}\n\nJudge feedback:\n{verdict.feedback}"
                ),
                role=node.role,
                depth=node.depth,
                parent_id=node.parent_id,
            )
            artifact = worker(retry_node, child_results)
            verdict = judge(node, artifact, child_results)

        return WorkArtifact(
            task_id=artifact.task_id,
            role=artifact.role,
            output=artifact.output,
            confidence=max(0.0, min(1.0, verdict.confidence)),
            findings=artifact.findings,
            evidence=artifact.evidence,
            open_questions=artifact.open_questions
            + ((verdict.feedback,) if verdict.feedback and not verdict.passed else ()),
            children=child_results,
        )

    def _solve_parallel(
        self,
        nodes: Sequence[TaskNode],
        *,
        decompose: Decomposer,
        worker: Worker,
        judge: Judge,
    ) -> tuple[WorkArtifact, ...]:
        with ThreadPoolExecutor(max_workers=self.budget.max_parallel) as pool:
            futures = [
                pool.submit(
                    self._solve,
                    node,
                    decompose=decompose,
                    worker=worker,
                    judge=judge,
                )
                for node in nodes
            ]
            return tuple(future.result() for future in futures)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    model_profile: str
    strengths: tuple[str, ...] = field(default_factory=tuple)
    max_retries: int = 1
