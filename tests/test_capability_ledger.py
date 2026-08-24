from pathlib import Path

from myai.capability_ledger import CapabilityLedger, CapabilityScore, CapabilitySnapshot


def test_capability_ledger_roundtrip_and_delta(tmp_path: Path) -> None:
    path = tmp_path / "capability.json"
    ledger = CapabilityLedger(path)
    first = CapabilitySnapshot("v8.0", "base", "2026-08-24T00:00:00Z", (CapabilityScore("coding", 0.70),), 1, 1)
    second = CapabilitySnapshot("v9.0", "head", "2026-08-24T00:01:00Z", (CapabilityScore("coding", 0.82),), 2, 2)
    ledger.record(first)
    ledger.record(second)
    restored = CapabilityLedger(path)
    assert restored.load() is True
    assert restored.current is not None
    assert restored.current.score("coding") == 0.82
    assert restored.delta(second, first)["coding"] == 0.12
    assert restored.regressions(second, first) == ()
    assert restored.baseline()["version"] == "v9.0"
