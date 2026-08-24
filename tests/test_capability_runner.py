from myai.capability_runner import run_architecture_benchmark


def test_architecture_benchmark_collects_executable_evidence() -> None:
    run = run_architecture_benchmark()

    assert run.elapsed_ms >= 0.0
    assert run.results["frontier-routing"][0] == 1.0
    assert run.results["recursive-agent-graph"][0] == 1.0
    assert run.results["frontier-routing"][1]
    assert run.results["recursive-agent-graph"][1]
