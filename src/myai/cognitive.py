from dataclasses import dataclass
from re import search
from typing import Literal

TaskKind = Literal["chat", "reasoning", "coding", "research", "memory"]


@dataclass(frozen=True)
class CognitivePlan:
    kind: TaskKind
    steps: tuple[str, ...]
    requires_retrieval: bool
    requires_verification: bool


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    score: float
    issues: tuple[str, ...]


class CognitiveCore:
    """Deterministic orchestration policy for MY-AI V6.

    The core does not replace the language model. It decides which cognitive
    steps should happen around the model: classify, retrieve, generate and
    verify. Keeping this layer model-agnostic lets OpenAI-compatible, local,
    and future native providers share the same safety/quality pipeline.
    """

    def classify(self, message: str) -> TaskKind:
        text = message.casefold().strip()
        if any(token in text for token in ("python", "code", "debug", "bug", "function", "api")):
            return "coding"
        if any(token in text for token in ("research", "sources", "cite", "latest", "compare")):
            return "research"
        if any(token in text for token in ("remember", "forget", "save this", "my preference")):
            return "memory"
        if any(token in text for token in ("why", "prove", "calculate", "derive", "analyze")):
            return "reasoning"
        return "chat"

    def plan(self, message: str, retrieved_count: int) -> CognitivePlan:
        kind = self.classify(message)
        requires_retrieval = kind in {"research", "memory", "reasoning"} or retrieved_count > 0
        requires_verification = kind in {"research", "reasoning", "coding"}

        steps: list[str] = ["classify"]
        if requires_retrieval:
            steps.append("retrieve")
        steps.extend(("generate", "verify" if requires_verification else "respond"))
        return CognitivePlan(
            kind=kind,
            steps=tuple(steps),
            requires_retrieval=requires_retrieval,
            requires_verification=requires_verification,
        )

    def verify(self, answer: str, retrieved_count: int) -> VerificationResult:
        text = answer.strip()
        issues: list[str] = []
        if not text:
            issues.append("empty_answer")
        if len(text) < 8:
            issues.append("answer_too_short")
        if search(r"\b(I|we)\s+(used|searched|verified)\b", text.casefold()) and retrieved_count == 0:
            issues.append("unsupported_tool_claim")

        score = max(0.0, 1.0 - 0.35 * len(issues))
        return VerificationResult(passed=not issues, score=score, issues=tuple(issues))
