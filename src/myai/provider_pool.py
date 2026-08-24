from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import httpx

from .config import Settings
from .schemas import ChatMessage

ModelTier = Literal["fast", "balanced", "frontier"]


@dataclass(frozen=True)
class ModelEndpoint:
    tier: ModelTier
    provider: str
    model: str
    base_url: str
    api_key: str
    timeout_seconds: float


class TieredModelPool:
    """Execute work on a tier-specific model endpoint with local fallback."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.endpoints = {
            "fast": ModelEndpoint(
                "fast",
                settings.fast_model_provider,
                settings.fast_model_name,
                settings.fast_model_base_url,
                settings.fast_model_api_key,
                settings.model_timeout_seconds,
            ),
            "balanced": ModelEndpoint(
                "balanced",
                settings.balanced_model_provider,
                settings.balanced_model_name,
                settings.balanced_model_base_url,
                settings.balanced_model_api_key,
                settings.model_timeout_seconds,
            ),
            "frontier": ModelEndpoint(
                "frontier",
                settings.frontier_model_provider,
                settings.frontier_model_name,
                settings.frontier_model_base_url,
                settings.frontier_model_api_key,
                settings.model_timeout_seconds,
            ),
        }

    def generate(self, tier: ModelTier, messages: Sequence[ChatMessage]) -> str:
        endpoint = self.endpoints[tier]
        if endpoint.provider.lower() in {"compatible", "openai-compatible", "openai_compatible"}:
            return self._generate_compatible(endpoint, messages)
        return self._fallback(endpoint, messages)

    @staticmethod
    def _generate_compatible(endpoint: ModelEndpoint, messages: Sequence[ChatMessage]) -> str:
        payload = {
            "model": endpoint.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": 0.2,
        }
        headers = {"Authorization": f"Bearer {endpoint.api_key}"} if endpoint.api_key else {}
        with httpx.Client(timeout=endpoint.timeout_seconds) as client:
            response = client.post(
                f"{endpoint.base_url.rstrip('/')}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"])

    @staticmethod
    def _fallback(endpoint: ModelEndpoint, messages: Sequence[ChatMessage]) -> str:
        user_message = next(
            (message.content for message in reversed(messages) if message.role == "user"),
            "",
        )
        return (
            f"MY-AI {endpoint.tier} fallback is active. Configure the {endpoint.tier} "
            f"model provider to run a real specialist model. "
            f"Task characters: {len(user_message.strip())}."
        )
