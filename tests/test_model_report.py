from myai import AIEngine, build_model_report


def test_model_report_is_derived_from_engine() -> None:
    engine = AIEngine()
    report = engine.model_report()
    assert report.version == engine.version
    assert "program_graph" in report.active_features
    assert "capability_benchmark" in report.active_features
    assert "tool_use" not in report.capability_baseline.get("scores", {}) or report.capability_baseline.get("scores", {}).get("tool_use") is None
    assert report.benchmark_cases
    assert report.compute_policies
    assert report.measured_claim_policy.startswith("Numeric capability claims require")


def test_public_builder_matches_engine_report() -> None:
    engine = AIEngine()
    direct = build_model_report(engine).to_dict()
    via_engine = engine.model_report().to_dict()
    assert direct == via_engine
