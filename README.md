# MY-AI

## V7.1 — Recursive Multi-Model Agent System

MY-AI V7.1 keeps the V5 semantic retrieval/persistent-memory foundation and V6 cognitive verification, then makes the V7 architecture executable across multiple model tiers.

### V7.1 architecture

```text
request
  -> cognitive classification
  -> adaptive tier selection
  -> recursive task graph
       -> specialist workers on fast/balanced/frontier models
       -> compact WorkArtifacts
       -> independent frontier judge
       -> retry only failed branch
  -> final verification
  -> persistent memory
```

The runtime now supports three separately configurable model tiers. Each tier can point to its own OpenAI-compatible endpoint/model, so a planner, specialist, critic, and judge can run on different models instead of pretending that one model did every job.

### Context engineering

Agents exchange structured artifacts instead of replaying their full conversation or repository context:

```text
WorkArtifact
  task_id
  role
  output
  confidence
  findings
  evidence
  open_questions
  children
```

This keeps detailed exploration local to the worker and sends only distilled work to the next stage.

### Recursive execution guarantees

Execution is bounded by maximum depth, node count, parallel workers, and retries. Cycles are detected. When a judge rejects a result, only that node is retried with judge feedback rather than restarting the whole graph.

### Code Intelligence

`CodeIntelligenceIndex` builds a lightweight Python AST/symbol graph. Coding tasks can retrieve a narrow set of relevant files, classes, functions, parents, and line ranges rather than repeatedly loading the entire repository.

```text
repository
  -> AST
  -> symbols + imports
  -> searchable code graph
  -> narrow context
  -> coding specialist
```

### Real model configuration

Set `MYAI_FAST_MODEL_PROVIDER`, `MYAI_BALANCED_MODEL_PROVIDER`, and `MYAI_FRONTIER_MODEL_PROVIDER` to `compatible` and provide the endpoint/model/key for each tier. Local mode remains deterministic and does not require a network model.

`MYAI_AGENT_MODE=auto` uses the recursive runtime for reasoning, research, and coding tasks; use `always` to force agent execution for every request.

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

Knowledge is persisted in `data/knowledge.sqlite3` by default. The `data/` directory is intentionally ignored by Git because it is runtime state.
