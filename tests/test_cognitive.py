from myai.cognitive import CognitiveCore


def test_classifies_task_types() -> None:
    core = CognitiveCore()
    assert core.classify("debug this Python API") == "coding"
    assert core.classify("compare the latest sources") == "research"
    assert core.classify("why does this happen?") == "reasoning"
    assert core.classify("hello there") == "chat"


def test_builds_retrieval_and_verification_plan() -> None:
    plan = CognitiveCore().plan("why is this true?", retrieved_count=2)
    assert plan.kind == "reasoning"
    assert plan.requires_retrieval is True
    assert plan.requires_verification is True
    assert plan.steps == ("classify", "retrieve", "generate", "verify")


def test_verification_rejects_unsupported_tool_claim() -> None:
    result = CognitiveCore().verify("I searched the web and found this.", retrieved_count=0)
    assert result.passed is False
    assert "unsupported_tool_claim" in result.issues
