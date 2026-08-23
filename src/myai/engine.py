from collections.abc import Sequence

from .config import get_settings
from .schemas import ChatMessage, ChatRequest, ChatResponse


class AIEngine:
    """Provider-agnostic V1 orchestration boundary.

    V1 deliberately uses a deterministic fallback so CI can exercise the
    complete request path without downloading model weights or requiring keys.
    A real local/remote model adapter can be plugged in behind this boundary.
    """

    version = "v1"

    def __init__(self) -> None:
        self.settings = get_settings()

    def generate(self, request: ChatRequest) -> ChatResponse:
        history = self._normalize_history(request.conversation)
        text = self._fallback_response(request.message, history)
        return ChatResponse(
            text=text,
            model=self.settings.model_name,
            version=self.version,
        )

    @staticmethod
    def _normalize_history(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        return [message for message in messages if message.content.strip()]

    @staticmethod
    def _fallback_response(message: str, history: Sequence[ChatMessage]) -> str:
        context_note = f" Context messages: {len(history)}." if history else ""
        return (
            "MY-AI V1 received your request and is ready for a real model adapter. "
            f"Your message has {len(message.strip())} characters.{context_note}"
        )
