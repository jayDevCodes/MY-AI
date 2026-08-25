from __future__ import annotations

from myai import AIEngine, V10AIEngine


def test_v10_is_public_engine() -> None:
    engine = AIEngine()
    assert isinstance(engine, V10AIEngine)
    assert engine.version == "v10.0"
    assert engine.settings.self_healing_enabled is True
    assert engine.inspection_mode("unknown_symbol") == "deep"


def test_v10_repair_context_uses_existing_cognitive_features() -> None:
    engine = AIEngine()
    traceback_text = (
        'Traceback (most recent call last):\n'
        '  File "src/myai/engine.py", line 1, in generate\n'
        "    raise TypeError('bad value')\n"
        'TypeError: bad value 123\n'
    )
    context = engine.repair_context_v10(traceback_text)
    rendered = "\n".join(message.content for message in context)
    assert "FAILURE SIGNATURE:" in rendered
    assert "CODE HEALTH MODE:" in rendered
    assert "COMPUTE POLICY:" in rendered
    assert "SOURCE CONTEXT:" in rendered
