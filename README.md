# MY-AI

## V7 — Recursive Agent Graph

MY-AI V7 keeps the V5 semantic retrieval/persistent-memory foundation and the V6 verification layer, then adds an adaptive hierarchical architecture for complex work.

### V7 architecture

```text
request
  -> cognitive classification
  -> adaptive model routing
  -> hierarchical task graph
       -> specialist branches in parallel
       -> structured WorkArtifacts
       -> local synthesis
       -> independent judge
       -> retry only failed node(s)
  -> final verification
  -> persistent memory
```

V7 deliberately avoids replaying a whole repository or conversation to every worker. Agents exchange compact structured artifacts containing output, findings, evidence, confidence, and open questions. Execution is bounded by depth, node, parallelism, and retry budgets to prevent runaway recursion.

### Adaptive model routing

`AdaptiveModelRouter` selects `fast`, `balanced`, or `frontier` tiers from task type, complexity, uncertainty, risk, context size, and latency requirements. The policy is model-vendor agnostic so different local or remote providers can be attached later.

### Code Intelligence

`CodeIntelligenceIndex` builds a lightweight Python AST/symbol index. Queries retrieve only relevant files, classes, functions, line ranges, and parents instead of repeatedly loading the complete repository. This provides the foundation for persistent project understanding and future code-graph traversal.

```text
repository
  -> AST
  -> symbols/imports
  -> searchable code graph
  -> narrow context for the coding agent
```

### Recursive execution

```text
Master Task
  ├─ Research Agent
  ├─ Coding Agent
  └─ Reasoning Agent
       ↓
structured artifacts
       ↓
Judge
  ├─ pass -> synthesize
  └─ fail -> retry only failed branch
```

The recursion is bounded and cycle-aware. Independent branches can run in parallel, while dependent synthesis happens after their artifacts are available.

### Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
cp .env.example .env
pytest
```

Runtime dependencies are listed in `requirements.txt`; development/CI dependencies are in `requirements-dev.txt`.

### Embeddings and knowledge

Sentence Transformers is the default embedding provider with `sentence-transformers/all-MiniLM-L6-v2`. The first local run downloads the embedding model into the normal model cache. `MYAI_EMBEDDING_DEVICE` supports `cpu`, `cuda`, `mps`, or automatic selection.

Knowledge is persisted in `data/knowledge.sqlite3` by default. The `data/` directory is intentionally ignored by Git because it is runtime state.

For CI and deterministic tests, set `MYAI_EMBEDDING_PROVIDER=deterministic` to avoid downloading model weights.

### V7 runtime controls

- `MYAI_COGNITIVE_VERIFICATION=true|false`
- `MYAI_COGNITIVE_MAX_RETRIES=1`
- `MYAI_AGENT_MAX_DEPTH=3`
- `MYAI_AGENT_MAX_NODES=32`
- `MYAI_AGENT_MAX_PARALLEL=4`
- `MYAI_AGENT_MAX_RETRIES=1`
- `MYAI_CODE_INDEX_ENABLED=true|false`
- `MYAI_CODE_INDEX_ROOT=.`
- `MYAI_CODE_CONTEXT_LIMIT=8`

The model provider remains pluggable through the existing provider interface. V7's orchestration layer is designed to combine local models, fast models, and frontier models rather than locking the system to one vendor.
