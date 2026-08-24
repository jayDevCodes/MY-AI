# MY-AI

## V6 — Cognitive Core

MY-AI V6 keeps the V5 semantic retrieval and persistent local knowledge foundation and adds a model-agnostic cognitive layer around generation.

### Cognitive pipeline

```text
request
  -> classify
  -> retrieve (when useful)
  -> generate
  -> verify
  -> bounded retry when verification fails
  -> persist conversation memory
```

The cognitive layer currently provides deterministic task classification for chat, reasoning, coding, research, and memory-oriented requests; execution planning; and lightweight answer verification. It does not pretend that a tool was used when no retrieval/tool result exists.

### Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
cp .env.example .env
pytest
```

Runtime dependencies are listed in `requirements.txt`; development/CI dependencies are in `requirements-dev.txt`.

### Semantic embeddings

Sentence Transformers is the default embedding provider with `sentence-transformers/all-MiniLM-L6-v2`. The first local run downloads the embedding model into the normal model cache. The embedding device can be selected with `MYAI_EMBEDDING_DEVICE` (`cpu`, `cuda`, `mps`, or blank for automatic selection).

For CI and deterministic tests, set `MYAI_EMBEDDING_PROVIDER=deterministic` to avoid downloading model weights.

### Knowledge database

Knowledge is persisted in `data/knowledge.sqlite3` by default. The `data/` directory is intentionally ignored by Git because it is runtime state, not source code.

### V6 runtime controls

- `MYAI_COGNITIVE_VERIFICATION=true|false` enables the verification stage.
- `MYAI_COGNITIVE_MAX_RETRIES=1` bounds automatic regeneration after a failed verification.

The model provider remains pluggable through the existing provider interface, so V6 can wrap local or OpenAI-compatible model servers without coupling the cognitive layer to one vendor.
