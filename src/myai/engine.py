from collections.abc import Sequence
from pathlib import Path
from time import perf_counter

from .agent_graph import ExecutionBudget
from .agent_runtime import MultiModelAgentRuntime
from .code_intelligence import CodeIntelligenceIndex
from .cognitive import CognitiveCore, CognitivePlan, VerificationResult
from .cognitive_state import Belief, CognitiveState, MemoryItem, MemoryKind
from .config import get_settings
from .context_contract import build_cognitive_context
from .embeddings import (
    DeterministicEmbeddingModel,
    EmbeddingModel,
    SentenceTransformerEmbeddingModel,
)
from .knowledge import Document, KnowledgeStore, RetrievedChunk, SQLiteVectorStore
from .memory import ConversationMemory
from .memory_lifecycle import MemoryLifecycleManager
from .memory_store import CognitiveMemoryStore
from .model_router import AdaptiveModelRouter, RoutingRequest, RoutingDecision
from .provider_pool import TieredModelPool
from .providers import get_provider
from .schemas import ChatMessage, ChatRequest, ChatResponse


class AIEngine:
    """V7.1 orchestration with multi-model agents, verification and code intelligence."""

    version = "v7.1"

    def __init__(self) -> None:
        self.settings = get_settings()
        self.provider = get_provider()
        self.memory = ConversationMemory()
        self.cognitive = CognitiveCore()
        self.cognitive_state = CognitiveState()
        self.memory_lifecycle = MemoryLifecycleManager()
        self.memory_store = CognitiveMemoryStore(self.settings.memory_store_path)
        self.memory_store.hydrate(self.cognitive_state, limit=self.settings.memory_load_limit)
        self.router = AdaptiveModelRouter()
        self.model_pool = TieredModelPool(self.settings)
        self.agent_runtime = MultiModelAgentRuntime(
            self.model_pool,
            self.router,
            ExecutionBudget(
                max_depth=self.settings.agent_max_depth,
                max_nodes=self.settings.agent_max_nodes,
                max_parallel=self.settings.agent_max_parallel,
                max_retries=self.settings.agent_max_retries,
            ),
        )
        self.knowledge = self._build_knowledge_store()
        self.code_index = CodeIntelligenceIndex()
        self._code_index_ready = self.code_index.load_snapshot(
            self.settings.code_index_snapshot_path,
            self.settings.code_index_root,
        )

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

    def refresh_code_index(self) -> int:
        """Rebuild and persist the project symbol graph after source changes."""
        self.code_index = CodeIntelligenceIndex()
        count = self.code_index.index_tree(self.settings.code_index_root)
        self.code_index.save_snapshot(
            self.settings.code_index_snapshot_path,
            self.settings.code_index_root,
        )
        self._code_index_ready = True
        return count

    def _ensure_code_index_fresh(self) -> None:
        if not self.settings.code_index_enabled:
            return
        if not self._code_index_ready:
            self._code_index_ready = self.code_index.load_snapshot(
                self.settings.code_index_snapshot_path,
                self.settings.code_index_root,
            )
        if not self._code_index_ready:
            self.refresh_code_index()
            return
        if self.code_index.refresh_if_stale(self.settings.code_index_root):
            self.code_index.save_snapshot(
                self.settings.code_index_snapshot_path,
                self.settings.code_index_root,
            )

    def code_context(self, query: str, *, limit: int | None = None) -> tuple[dict[str, object], ...]:
        """Return only relevant symbols instead of repeatedly loading the whole repository."""
        if not self.settings.code_index_enabled:
            return ()
        self._ensure_code_index_fresh()
        return self.code_index.context_map(
            query,
            limit=limit if limit is not None else self.settings.code_context_limit,
        )

    def code_source_context(self, query: str, *, limit: int = 5) -> tuple[dict[str, object], ...]:
        source_context = self.code_index.read_context(query, limit=limit) if self.code_context(query, limit=limit) else ()
        return source_context

    def route(self, request: ChatRequest, retrieved_count: int = 0) -> RoutingDecision:
        plan = self.cognitive.plan(request.message, retrieved_count)
        return self.router.choose(
            RoutingRequest(
                task_kind=plan.kind,
                complexity=min(
                    1.0,
                    0.35 + 0.1 * len(plan.steps) + 0.05 * len(request.message) / 1000,
                ),
                uncertainty=0.6 if plan.requires_verification else 0.2,
                context_size=sum(len(message.content) for message in request.conversation),
                risk="high" if plan.kind in {"research", "coding"} else "low",
                latency_sensitive=False,
                quality_priority=True,
            )
        )

    def run_agent_task(self, request: ChatRequest, retrieved: Sequence[RetrievedChunk] = ()) -> str:
        plan = self.cognitive.plan(request.message, len(retrieved))
        context = self._build_context_messages(request.message, request.conversation, retrieved, plan)
        return self.agent_runtime.run(
            request.message,
            plan.kind,
            context,
            state=self.cognitive_state,
        ).artifact.output

    def _proven_strategy_hints(self) -> tuple[str, ...]:
        provider = getattr(self, "strategy_scores", None)
        if not callable(provider):
            return ()
        try:
            return tuple(
                f"{item.strategy} (success={item.success_rate:.2f}, score={item.mean_score:.2f})"
                for item in provider(limit=3)
            )
        except (AttributeError, TypeError, ValueError):
            return ()

    def _record_generation_experience(
        self,
        *,
        task: str,
        strategy: str,
        verification: VerificationResult,
        latency_ms: float,
    ) -> None:
        recorder = getattr(self, "record_generation_experience", None)
        if callable(recorder):
            recorder(
                task=task,
                strategy=strategy,
                success=verification.passed,
                score=verification.score,
                latency_ms=latency_ms,
            )

    def _consolidate_memory_if_due(self) -> None:
        """Promote repeated verified episodes without mutating raw evidence."""
        if len(self.cognitive_state.memories) == 0 or len(self.cognitive_state.memories) % 16 != 0:
            return
        existing = {(item.kind, item.content) for item in self.cognitive_state.memories}
        for promotion in self.memory_lifecycle.consolidate(self.cognitive_state.memories):
            if (promotion.kind, promotion.content) not in existing:
                self.cognitive_state.add_memory(promotion)
                existing.add((promotion.kind, promotion.content))

    def generate(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        history = self._normalize_history(request.conversation)
        retrieved = self.retrieve(request.message)
        plan = self.cognitive.plan(request.message, len(retrieved))
        routing = self.route(request, len(retrieved))

        self.cognitive_state.set_goal(request.message.strip(), subgoals=list(plan.steps))
        self.cognitive_state.active_strategy = f"route:{routing.tier}:{plan.kind}"
        self.cognitive_state.set_uncertainty(0.6 if plan.requires_verification else 0.25)
        self.cognitive_state.observe(f"retrieved:{len(retrieved)}")
        self.cognitive_state.add_belief(
            Belief(
                statement=f"Task classified as {plan.kind}",
                confidence=max(0.0, min(1.0, 1.0 - self.cognitive_state.uncertainty)),
                provenance=("cognitive-plan",),
            )
        )
        for item in retrieved:
            self.cognitive_state.add_memory(
                MemoryItem(
                    content=item.text,
                    kind=MemoryKind.SEMANTIC,
                    importance=max(0.0, min(1.0, item.score)),
                    confidence=max(0.0, min(1.0, item.score)),
                    provenance=(item.source,),
                    tags=("retrieved", plan.kind),
                )
            )

        mode = self.settings.agent_mode.lower()
        use_agents = mode == "always" or (mode == "auto" and plan.kind in {"research", "reasoning", "coding"})

        if use_agents:
            text = self.run_agent_task(request, retrieved)
        else:
            text = self.provider.generate(self._build_messages(request.message, history, retrieved, plan, routing))

        verification = VerificationResult(passed=True, score=1.0, issues=())
        if self.settings.cognitive_verification:
            verification = self.cognitive.verify(text, len(retrieved))
            self.cognitive_state.set_uncertainty(max(0.0, min(1.0, 1.0 - verification.score)))
            self.cognitive_state.observe(
                f"verification:{'passed' if verification.passed else 'failed'}:{verification.score:.2f}"
            )
            if not verification.passed and self.settings.cognitive_max_retries > 0:
                retry_instruction = ChatMessage(
                    role="system",
                    content=(
                        "Verification found answer-quality issues: "
                        f"{', '.join(verification.issues)}. Regenerate a concise, honest answer."
                    ),
                )
                if use_agents:
                    retry_context = self._build_context_messages(
                        request.message,
                        request.conversation,
                        retrieved,
                        plan,
                    )
                    retry_context.append(retry_instruction)
                    text = self.agent_runtime.run(
                        request.message,
                        plan.kind,
                        retry_context,
                        state=self.cognitive_state,
                    ).artifact.output
                else:
                    messages = self._build_messages(request.message, history, retrieved, plan, routing)
                    text = self.provider.generate([*messages, retry_instruction])
                verification = self.cognitive.verify(text, len(retrieved))
                self.cognitive_state.set_uncertainty(max(0.0, min(1.0, 1.0 - verification.score)))

        self.cognitive_state.add_memory(
            MemoryItem(
                content=text,
                kind=MemoryKind.EPISODIC,
                importance=0.65,
                confidence=max(0.0, min(1.0, verification.score)),
                provenance=("generation",),
                tags=(plan.kind, routing.tier),
            )
        )
        self._consolidate_memory_if_due()
        self.memory_store.persist_state(self.cognitive_state)
        self._record_generation_experience(
            task=request.message,
            strategy=self.cognitive_state.active_strategy or f"route:{routing.tier}:{plan.kind}",
            verification=verification,
            latency_ms=(perf_counter() - started) * 1000.0,
        )
        self.memory.extend(history)
        self.memory.add(ChatMessage(role="user", content=request.message.strip()))
        self.memory.add(ChatMessage(role="assistant", content=text))

        return ChatResponse(text=text, model=self.settings.model_name, version=self.version)

    @staticmethod
    def _normalize_history(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
        return [message for message in messages if message.content.strip()]

    def _build_context_messages(
        self,
        message: str,
        history: Sequence[ChatMessage],
        retrieved: Sequence[RetrievedChunk] = (),
        plan: CognitivePlan | None = None,
    ) -> list[ChatMessage]:
        messages = [ChatMessage(role="system", content=self.settings.system_prompt)]
        context = build_cognitive_context(
            self.cognitive_state,
            query=message,
            memory_kinds=(
                MemoryKind.FAILURE,
                MemoryKind.PROCEDURAL,
                MemoryKind.STRATEGIC,
                MemoryKind.SEMANTIC,
                MemoryKind.EPISODIC,
            ),
            memory_limit=6,
            belief_limit=6,
            observation_limit=6,
            capability_limit=8,
            world_limit=8,
            strategy_hints=self._proven_strategy_hints(),
        )
        rendered_context = context.render()
        if rendered_context:
            messages.append(
                ChatMessage(
                    role="system",
                    content=(
                        "Bounded shared cognitive context. Use it as evidence-aware context, not ground truth; "
                        "distinguish beliefs, memories and observations. Proven strategies are historical signals, not guarantees.\n"
                        + rendered_context
                    ),
                )
            )
        if plan is not None:
            messages.append(
                ChatMessage(
                    role="system",
                    content=f"Cognitive task: {plan.kind}. Execution stages: {', '.join(plan.steps)}.",
                )
            )
        if retrieved:
            context_lines = [
                "Retrieved semantic knowledge context. Treat it as reference material and do not invent sources:"
            ]
            context_lines.extend(
                f"[{item.source}#{item.chunk_index} score={item.score:.4f}] {item.text}"
                for item in retrieved
            )
            messages.append(ChatMessage(role="system", content="\n".join(context_lines)))
        if plan is not None and plan.kind == "coding":
            source_context = self.code_source_context(message, limit=min(self.settings.code_context_limit, 5))
            if source_context:
                messages.append(
                    ChatMessage(
                        role="system",
                        content=(
                            "Relevant repository source slices. Modify only after inspecting these exact ranges:\n"
                            + "\n\n".join(
                                f"[{item['path']}:{item['start_line']}-{item['end_line']}]\n{item['text']}"
                                for item in source_context
                            )
                        ),
                    )
                )
        messages.extend(history[-8:])
        messages.append(ChatMessage(role="user", content=message.strip()))
        return messages

    def _build_messages(
        self,
        message: str,
        history: Sequence[ChatMessage],
        retrieved: Sequence[RetrievedChunk] = (),
        plan: CognitivePlan | None = None,
        routing: RoutingDecision | None = None,
    ) -> list[ChatMessage]:
        messages = self._build_context_messages(message, history, retrieved, plan)
        if routing is not None:
            messages.insert(
                min(2, len(messages)),
                ChatMessage(
                    role="system",
                    content=(
                        f"Routing tier: {routing.tier}. {routing.reason}. "
                        f"Parallel work allowed: {routing.allow_parallel}."
                    ),
                )
            )
        return messages
