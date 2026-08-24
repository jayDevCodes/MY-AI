from __future__ import annotations

from pathlib import Path

from myai import (
    CodeHealth,
    CodeHealthStore,
    FaultInjectionLab,
    SelfHealingRuntime,
    FailureSignatureStore,
)


def test_failure_signature_reuse_and_verified_episode(tmp_path: Path) -> None:
    signatures = FailureSignatureStore(tmp_path / "signatures.jsonl")
    runtime = SelfHealingRuntime(
        episode_path=tmp_path / "episodes.jsonl",
        signature_store=signatures,
    )
    signature = runtime.signature("TypeError", "bad value 123", "refresh_token")

    episode = runtime.verified_repair(
        signature=signature,
        reproduce=lambda: (True, "reproduced"),
        validate=lambda: True,
        lesson="validated minimal fix",
    )

    assert episode.status == "verified"
    assert signatures.similar(runtime.signature("TypeError", "bad value 456", "refresh_token"))


def test_fault_lab_restores_invariant() -> None:
    state = {"healthy": True}
    lab = FaultInjectionLab()
    result = lab.run(
        lab.simple_toggle(state, "healthy"),
        detector=lambda: state["healthy"] is False,
    )
    assert result.detected is True
    assert result.recovered is True
    assert result.verified is True
    assert state["healthy"] is True


def test_code_health_prefers_reuse_for_stable_verified_symbol(tmp_path: Path) -> None:
    store = CodeHealthStore(tmp_path / "health.json")
    store.upsert(
        CodeHealth(
            symbol="stable_fn",
            stability=0.99,
            verification_confidence=0.99,
            failure_rate=0.0,
            change_frequency=0.01,
            dependency_centrality=0.05,
            last_verified="2026-08-25T00:00:00+00:00",
        )
    )
    assert store.inspection_mode("stable_fn") == "reuse"
