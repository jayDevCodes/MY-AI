from __future__ import annotations

from myai import AIEngine, V10AIEngine


def test_v10_public_engine_and_repair_policy() -> None:
    engine = AIEngine()
    assert isinstance(engine, V10AIEngine)
    assert engine.version == "v10.0"
    policy = engine.repair_compute_policy(
        "Traceback (most recent call last):\n"
        "  File 'src/myai/engine.py', line 10, in generate\n"
        "    value = session['id']\n"
        "TypeError: NoneType is not subscriptable"
    )
    assert policy.preferred_tier in {"fast", "balanced", "frontier"}
    assert 1 <= policy.reasoning_depth <= 3
    assert 1 <= policy.verification_passes <= 2


def test_v10_repair_context_contains_operational_guards() -> None:
    engine = AIEngine()
    context = engine.repair_context_v10(
        "Traceback (most recent call last):\n"
        "  File 'src/myai/engine.py', line 10, in generate\n"
        "TypeError: NoneType is not subscriptable"
    )
    rendered = "\n".join(message.content for message in context)
    assert "FAILURE SIGNATURE:" in rendered
    assert "CODE HEALTH INSPECTION MODE:" in rendered
    assert "COMPUTE POLICY:" in rendered
    assert "validation-gated" in rendered


def test_v10_unknown_symbols_require_deep_inspection() -> None:
    engine = AIEngine()
    assert engine.inspection_mode("unknown_symbol") == "deep"
