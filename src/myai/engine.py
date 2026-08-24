from collections.abc import Sequence
from pathlib import Path

from .config import get_settings
from .embeddings import (
    DeterministicEmbeddingModel,
    EmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from .knowledge import Document, KnowledgeStore, RetrievedChunk, SQLiteVectorStore
from .memory import ConversationMemory
from .providers import get_provider
from .schemas import ChatMessage, ChatRequest, ChatResponse


class AIEngine:
    """V5 orchestration layer with semantic retrieval and persistent knowledge."""

    version = "v5"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = get_provider()
        self.memory = ConversationMemory()
        self.knowledge = self._build_knowledge_store()

    def _build_knowledge_store(self) -> KnowledgeStore:
        if self.settings.embedding_provider.lower() == "deterministic":
            embedder: EmbeddingModel = DeterministicEmbeddingModel()
        else:
            embedder = SentenceTransformerEmbeddingModel(
                self.settings.embedding_model_name,
                device=self.settings.embedding_device,
            )
        return SQLiteVectorStore(Path(self.settings.knowledge_db_path), embedder)

    def add_document(self, document: Document) -> int:
        return self.knowledge.add(
            document,
            chunk_size=self.settings.knowledge_chunk_size,
            overlap=self.settings.knowledge_chunk_overlap,
        )

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedChunk]:
        return self.knowledge.search(
            query,
            top_k=top_k if top_k is not None else self.settings.knowledge_top_k,
        )

    def generate(self, request: ChatRequest) -> ChatResponse:
        history = self._normalize_history(request.conversation)
        retrieved = self.retrieve(request.message)
        messages = self._build_messages(request.message, history, retrieved)
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
        self,
        message: str,
        history: Sequence[ChatMessage],
        retrieved: Sequence[RetrievedChunk] = (),
    ) -> list[ChatMessage]:
        system = ChatMessage(role="system", content=self.settings.system_prompt)
        context_messages: list[ChatMessage] = []
        if retrieved:
            context_lines = [
                "Retrieved semantic knowledge context. Treat it as reference material "
                "and do not invent sources:"
            ]
            for item in retrieved:
                context_lines.append(
                    f"[{item.source}#{item.chunk_index} score={item.score:.4f}] {item.text}"
                )
            context_messages.append(ChatMessage(role="system", content="\n".join(context_lines)))

        current = ChatMessage(role="user", content=message.strip())
        return [system, *context_messages, *history, current]
