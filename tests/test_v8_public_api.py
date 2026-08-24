from pathlib import Path

from myai import AIEngine, LegacyAIEngine, V8AIEngine
from myai.repository_twin import CausalRepositoryTwin


def test_public_engine_defaults_to_v8() -> None:
    assert AIEngine is V8AIEngine
    assert LegacyAIEngine.version == "v7.1"
    assert V8AIEngine.version == "v8.0"


def test_v8_engine_exposes_repository_twin(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MYAI_EMBEDDING_PROVIDER", "deterministic")
    monkeypatch.setenv("MYAI_CODE_INDEX_ROOT", str(tmp_path))
    monkeypatch.setenv("MYAI_CODE_INDEX_SNAPSHOT_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("MYAI_REPAIR_MEMORY_PATH", str(tmp_path / "repair.jsonl"))
    engine = V8AIEngine()
    assert isinstance(engine.repository_twin, CausalRepositoryTwin)

    source = tmp_path / "app.py"
    source.write_text("def refresh_token():\n    return None\n", encoding="utf-8")
    engine.refresh_repository_twin()
    assert engine.code_context("refresh token")

    traceback_text = (
        f'Traceback (most recent call last):\n  File "{source}", line 2, in refresh_token\n'
        "    return None['token']\nTypeError: 'NoneType' object is not subscriptable\n"
    )
    diagnosis = engine.diagnose_failure(traceback_text)
    assert diagnosis.error_type == "TypeError"
    assert diagnosis.primary_frame is not None
