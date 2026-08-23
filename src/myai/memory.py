from collections.abc import Sequence

from .config import get_settings
from .schemas import ChatMessage


class ConversationMemory:
    """Bounded in-process conversation memory for V2."""

    def __init__(self, max_messages: int | None = None) -> None:
        configured = get_settings().memory_max_messages
        self.max_messages = max_messages or configured
        self._messages: list[ChatMessage] = []

    def add(self, message: ChatMessage) -> None:
        self._messages.append(message)
        self._messages = self._messages[-self.max_messages :]

    def extend(self, messages: Sequence[ChatMessage]) -> None:
        for message in messages:
            self.add(message)

    def messages(self) -> list[ChatMessage]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages.clear()
