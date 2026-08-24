from myai.capability_runner import run_architecture_benchmark


def test_expanded_capability_runner_covers_reported_dimensions() -> None:
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
    assert expected.issubset(run.results)
    assert all(0.0 <= score <= 1.0 for score, _ in run.results.values())
    assert run.elapsed_ms >= 0.0
