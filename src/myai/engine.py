from collections.abc import Sequence

from .config import get_settings
from .memory import ConversationMemory
from .providers import get_provider
from .schemas import ChatMessage, ChatRequest, ChatResponse


class AIEngine:
    """V2 orchestration layer with bounded memory and pluggable inference."""

    version = "v2"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = get_provider()
        self.memory = ConversationMemory()

    def generate(self, request: ChatRequest) -> ChatResponse:
        history = self._normalize_history(request.conversation)
        messages = self._build_messages(request.message, history)
        text = self.provider.generate(messages)

        self.memory.extend(history)
        self.memory.add(ChatMessage(role="user", content=request.message.strip()))
        self.memory.add(ChatMessage(role="assistant", content=text))

        return ChatResponse(
            text=text,
            model=self.settings.model_name,
            version=self.version,
        )

    @staticmethod
    def _normalize_history(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        return [message for message in messages if message.content.strip()]

    def _build_messages(
        self, message: str, history: Sequence[ChatMessage]
    ) -> list[ChatMessage]:
        system = ChatMessage(role="system", content=self.settings.system_prompt)
        current = ChatMessage(role="user", content=message.strip())
        return [system, *history, current]
