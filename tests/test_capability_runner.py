from myai.capability_runner import run_architecture_benchmark


def test_architecture_benchmark_collects_executable_evidence() -> None:
    run = run_architecture_benchmark()

    expected = {
        "frontier-routing",
        "recursive-agent-graph",
        "cross-episode-repair-memory",
        "trace-verification",
        "program-graph-localization",
        "causal-trace-diagnosis",
        "targeted-context",
        "strategy-promotion",
    }
    assert run.elapsed_ms >= 0.0
    assert expected.issubset(run.results)
    assert all(0.0 <= score <= 1.0 for score, _ in run.results.values())
    assert all(run.results[name][1] for name in expected)
