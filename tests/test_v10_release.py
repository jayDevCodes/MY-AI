from __future__ import annotations

from myai import AIEngine, V10AIEngine


def test_v10_is_public_engine() -> None:
    engine = AIEngine()
    assert isinstance(engine, V10AIEngine)
    assert engine.version == "v10.0"
    assert engine.settings.self_healing_enabled is True
    assert engine.inspection_mode("unknown_symbol") == "deep"
