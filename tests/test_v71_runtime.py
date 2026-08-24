from pathlib import Path

from myai.agent_graph import ExecutionBudget
from myai.agent_runtime import MultiModelAgentRuntime
from myai.code_intelligence import CodeIntelligenceIndex
from myai.model_router import AdaptiveModelRouter
from myai.provider_pool import ModelTier
from myai.schemas import ChatMessage


class FakePool:
    def __init__(self) -> None:
        self.calls: list[tuple[ModelTier, str]] = []

    def generate(self, tier: ModelTier, messages: list[ChatMessage]) -> str:
        prompt = messages[-1].content
        self.calls.append((tier, prompt))
        if tier == "frontier":
            return '{"passed":true,"confidence":0.95,"feedback":"consistent"}'
        return f"worker:{tier}:{prompt[:40]}"


def test_runtime_executes_multiple_tiers_and_frontier_judge() -> None:
    pool = FakePool()
    runtime = MultiModelAgentRuntime(
        pool,
        AdaptiveModelRouter(),
        ExecutionBudget(max_depth=2, max_nodes=8, max_parallel=3, max_retries=1),
    )

    result = runtime.run(
        "design a robust API change",
        "coding",
        [ChatMessage(role="system", content="Use only supplied context.")],
    )

    tiers = {tier for tier, _ in pool.calls}
    assert "frontier" in tiers
    assert "balanced" in tiers or "fast" in tiers
    assert result.artifact.confidence == 0.95
    assert result.tier_by_task


def test_runtime_preserves_compact_artifact_handoff() -> None:
    pool = FakePool()
    runtime = MultiModelAgentRuntime(
        pool,
        AdaptiveModelRouter(),
        ExecutionBudget(max_depth=2, max_nodes=8, max_parallel=2, max_retries=1),
    )
    result = runtime.run("compare two approaches", "reasoning")
    assert result.artifact.children
    assert all(child.findings for child in result.artifact.children)


def test_code_index_snapshot_roundtrip_and_invalidation(tmp_path: Path) -> None:
    source = tmp_path / "sample.py"
    source.write_text("def refresh_token():\n    return True\n", encoding="utf-8")
    snapshot = tmp_path / "index.json"

    original = CodeIntelligenceIndex()
    assert original.index_tree(tmp_path) == 1
    original.save_snapshot(snapshot, tmp_path)

    restored = CodeIntelligenceIndex()
    assert restored.load_snapshot(snapshot, tmp_path) is True
    context = restored.context_map("refresh token")
    assert context
    assert context[0]["symbol"] == "refresh_token"

    source.write_text("def refresh_token():\n    return 'NEW_TOKEN'\n", encoding="utf-8")
    assert restored.load_snapshot(snapshot, tmp_path) is False
    assert restored.refresh_if_stale(tmp_path) is True
    refreshed = restored.read_context("refresh token")
    assert refreshed
    assert "NEW_TOKEN" in refreshed[0]["text"]
