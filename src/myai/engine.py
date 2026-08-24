from collections.abc import Sequence
from pathlib import Path

from .code_intelligence import CodeIntelligenceIndex
from .cognitive import CognitiveCore, CognitivePlan, VerificationResult
from .config import get_settings
from .embeddings import (
    DeterministicEmbeddingModel,
    EmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from .knowledge import Document, KnowledgeStore, RetrievedChunk, SQLiteVectorStore
from .memory import ConversationMemory
from .model_router import AdaptiveModelRouter, RoutingRequest, RoutingDecision
from .providers import get_provider
from .schemas import ChatMessage, ChatRequest, ChatResponse


class AIEngine:
    """V7 orchestration layer with routing, recursive-agent primitives and code intelligence."""

    version = "v7"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = get_provider()
        self.memory = ConversationMemory()
        self.cognitive = CognitiveCore()
        self.router = AdaptiveModelRouter()
        self.knowledge = self._build_knowledge_store()
        self.code_index = CodeIntelligenceIndex()
        self._code_index_ready = False

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

    def code_context(self, query: str, *, limit: int | None = None) -> tuple[dict[str, object], ...]:
        """Return only relevant symbols instead of repeatedly loading the whole repository."""
        if not self.settings.code_index_enabled:
            return ()
        if not self._code_index_ready:
            self.code_index.index_tree(self.settings.code_index_root)
            self._code_index_ready = True
        return self.code_index.context_map(
            query,
            limit=limit if limit is not None else self.settings.code_context_limit,
        )

    def route(self, request: ChatRequest, retrieved_count: int = 0) -> RoutingDecision:
        plan = self.cognitive.plan(request.message, retrieved_count)
        return self.router.choose(
            RoutingRequest(
                task_kind=plan.kind,
                complexity=min(1.0, 0.35 + 0.1 * len(plan.steps) + 0.05 * len(request.message) / 1000),
                uncertainty=0.6 if plan.requires_verification else 0.2,
                context_size=sum(len(message.content) for message in request.conversation),
                risk="high" if plan.kind in {"research", "coding"} else "low",
                latency_sensitive=False,
                quality_priority=True,
            )
        )

    def generate(self, request: ChatRequest) -> ChatResponse:
        history = self._normalize_history(request.conversation)
        retrieved = self.retrieve(request.message)
        plan = self.cognitive.plan(request.message, len(retrieved))
        routing = self.route(request, len(retrieved))
        messages = self._build_messages(request.message, history, retrieved, plan, routing)
        text = self.provider.generate(messages)

        verification = VerificationResult(passed=True, score=1.0, issues=())
        if self.settings.cognitive_verification:
            verification = self.cognitive.verify(text, len(retrieved))
            if not verification.passed and self.settings.cognitive_max_retries > 0:
                retry_instruction = ChatMessage(
                    role="system",
                    content=(
                        "Verification found answer-quality issues: "
                        f"{', '.join(verification.issues)}. Regenerate a concise, honest answer."
                    ),
                )
                text = self.provider.generate([*messages, retry_instruction])
                verification = self.cognitive.verify(text, len(retrieved))

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
        plan: CognitivePlan | None = None,
        routing: RoutingDecision | None = None,
    ) -> list[ChatMessage]:
        system = ChatMessage(role="system", content=self.settings.system_prompt)
        context_messages: list[ChatMessage] = []
        if plan is not None:
            context_messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        f"Cognitive task: {plan.kind}. "
                        f"Execution stages: {', '.join(plan.steps)}."
                    ),
                )
            )
        if routing is not None:
            context_messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        f"Routing tier: {routing.tier}. {routing.reason}. "
                        f"Parallel work allowed: {routing.allow_parallel}."
                    ),
                )
            )
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
