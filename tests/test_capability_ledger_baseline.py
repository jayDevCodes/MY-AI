from myai.capability_benchmark import CapabilityBenchmark
from myai.capability_ledger import CapabilityLedger


def test_unmeasured_capabilities_are_null_in_baseline(tmp_path) -> None:
    results = {
        "frontier-routing": (1.0, ("routing evidence",)),
        "recursive-agent-graph": (1.0, ("graph evidence",)),
    }
    snapshot = CapabilityBenchmark().snapshot(
        version="v10",
        commit="test",
        results=results,
    )
    ledger = CapabilityLedger(tmp_path / "capabilities.json")
    ledger.record(snapshot)

    baseline = ledger.baseline()

    assert baseline["scores"]["reasoning"] == 1.0
    assert baseline["scores"]["planning"] == 1.0
    assert baseline["scores"]["tool_use"] is None
    assert baseline["scores"]["coding"] is None
