from collections.abc import Sequence
from typing import Protocol

import httpx

from .config import get_settings
from .schemas import ChatMessage


class ModelProvider(Protocol):
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        """Generate assistant text from a normalized conversation."""


class FallbackProvider:
    def generate(self, messages: Sequence[ChatMessage]) -> str:
        user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return (
            "MY-AI V2 fallback is active. Configure MYAI_MODEL_PROVIDER=compatible "
            "and MYAI_MODEL_BASE_URL to connect a real model. "
            f"Your message has {len(user_message.strip())} characters."
        )


class OpenAICompatibleProvider:
    """Small OpenAI-compatible HTTP adapter for local or remote inference servers."""

    def __init__(self) -> None:
        settings = get_settings()
        self.base_url = settings.model_base_url.rstrip("/")
        self.model_name = settings.model_name
        self.api_key = settings.model_api_key
        self.timeout = settings.model_timeout_seconds

    def generate(self, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": self.model_name,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])


def get_provider() -> ModelProvider:
    provider = get_settings().model_provider.lower()
    if provider in {"compatible", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider()
    return FallbackProvider()
