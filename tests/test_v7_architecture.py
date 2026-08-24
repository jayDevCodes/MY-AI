from pathlib import Path

from myai.agent_graph import ExecutionBudget, RecursiveAgentGraph, TaskNode, WorkArtifact
from myai.code_intelligence import CodeIntelligenceIndex
from myai.model_router import AdaptiveModelRouter, RoutingRequest


def test_recursive_graph_parallel_synthesis_and_retry() -> None:
    calls: list[str] = []

    def decompose(node: TaskNode, budget: ExecutionBudget):
        if node.depth == 0:
            return (
                TaskNode("a", "collect facts", "research", 1, node.id),
                TaskNode("b", "implement logic", "coding", 1, node.id),
            )
        return ()

    def worker(node: TaskNode, children):
        calls.append(node.id)
        return WorkArtifact(node.id, node.role, f"done:{node.objective}", 0.8)

    attempts = {"root": 0}

    def judge(node: TaskNode, artifact: WorkArtifact, children):
        if node.id == "root" and attempts["root"] == 0:
            attempts["root"] += 1
            from myai.agent_graph import JudgeVerdict

            return JudgeVerdict(False, 0.2, "synthesis needs another pass")
        from myai.agent_graph import JudgeVerdict

        return JudgeVerdict(True, 0.9)

    root = TaskNode("root", "solve system task")
    result = RecursiveAgentGraph(ExecutionBudget(max_depth=2, max_nodes=8, max_parallel=2, max_retries=1)).run(
        root, decompose=decompose, worker=worker, judge=judge
    )
    assert result.confidence == 0.9
    assert {"a", "b", "root"}.issubset(set(calls))
    assert attempts["root"] == 1


def test_router_escalates_high_uncertainty_research() -> None:
    decision = AdaptiveModelRouter().choose(
        RoutingRequest(task_kind="research", complexity=0.9, uncertainty=0.8)
    )
    assert decision.tier == "frontier"
    assert decision.allow_parallel is True


def test_code_index_returns_narrow_symbol_context(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text(
        "class AuthService:\n"
        "    def refresh_token(self):\n"
        "        return True\n\n"
        "def health():\n"
        "    return 'ok'\n",
        encoding="utf-8",
    )
    index = CodeIntelligenceIndex()
    assert index.index_tree(tmp_path) == 1
    result = index.context_map("refresh token")
    assert result
    assert result[0]["symbol"] == "refresh_token"
