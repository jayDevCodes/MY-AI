from pathlib import Path

from myai.code_intelligence import CodeIntelligenceIndex
from myai.repository_twin import CausalRepositoryTwin
from myai.self_healing import CausalErrorEngine, RepairMemory, RepairMemoryRecord


def test_causal_diagnosis_uses_traceback_and_impact_slice(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    (package / "db.py").write_text(
        "def load_session():\n    return None\n",
        encoding="utf-8",
    )
    (package / "auth.py").write_text(
        "from pkg.db import load_session\n\ndef refresh_token():\n    session = load_session()\n    return session['token']\n",
        encoding="utf-8",
    )

    index = CodeIntelligenceIndex()
    assert index.index_tree(tmp_path) == 2
    twin = CausalRepositoryTwin(index)
    twin.rebuild(tmp_path)
    memory = RepairMemory(tmp_path / "repair_memory.jsonl")
    engine = CausalErrorEngine(twin, memory)

    traceback_text = (
        'Traceback (most recent call last):\n'
        f'  File "{package / "auth.py"}", line 5, in refresh_token\n'
        "    return session['token']\n"
        "TypeError: 'NoneType' object is not subscriptable\n"
    )
    diagnosis = engine.diagnose(traceback_text)

    assert diagnosis.error_type == "TypeError"
    assert diagnosis.primary_frame is not None
    assert diagnosis.primary_frame.symbol == "refresh_token"
    assert str(package / "auth.py") in diagnosis.affected_files
    assert diagnosis.impact.source_context
    assert diagnosis.confidence >= 0.8


def test_repair_memory_reuses_successful_history(tmp_path: Path) -> None:
    memory = RepairMemory(tmp_path / "memory.jsonl")
    memory.append(
        RepairMemoryRecord(
            error_type="TypeError",
            signature="typeerror: 'nonetype' object is not subscriptable",
            root_cause="session initialization returned None",
            patch_summary="guard session before token access",
            validation="unit tests + regression test",
            success=True,
            timestamp="2026-08-24T10:00:00+00:00",
        )
    )
    matches = memory.similar("TypeError", "'NoneType' object is not subscriptable")
    assert matches
    assert matches[0].patch_summary == "guard session before token access"
