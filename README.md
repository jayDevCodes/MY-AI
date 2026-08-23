# MY-AI

## V4

MY-AI V4 adds semantic retrieval with persistent local knowledge storage.

### Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
cp .env.example .env
pytest
```

The runtime dependencies are listed in `requirements.txt`; development/CI dependencies are in `requirements-dev.txt`.

### Semantic embeddings

V4 uses Sentence Transformers by default with `sentence-transformers/all-MiniLM-L6-v2`. The first local run downloads the embedding model into the normal model cache. The embedding device can be selected with `MYAI_EMBEDDING_DEVICE` (`cpu`, `cuda`, `mps`, or blank for automatic selection).

### Knowledge database

Knowledge is persisted in `data/knowledge.sqlite3` by default. The `data/` directory is intentionally ignored by Git because it is runtime state, not source code.

For CI and deterministic tests, set `MYAI_EMBEDDING_PROVIDER=deterministic` to avoid downloading model weights.
