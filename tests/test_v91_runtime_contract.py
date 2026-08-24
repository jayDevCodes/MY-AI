from myai import CognitiveState, V9AIEngine


def test_v91_public_runtime_contract() -> None:
    engine = V9AIEngine
    state = CognitiveState()
    assert state.uncertainty == 0.5
    assert engine.version == "v9.1"
