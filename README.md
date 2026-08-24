# MY-AI

## V8 — Causal Repository Twin + Self-Healing Agents

MY-AI V8 keeps the V7.1 recursive multi-model architecture and adds an evidence-first repository intelligence loop for targeted debugging and repair.

### V8 architecture

```text
runtime failure
  -> traceback + stack frames
  -> causal repository twin
       -> repository structure
       -> import/dependency impact
       -> symbol/source slice
       -> verification history
       -> previous successful fixes
  -> compact repair context
  -> specialist model
  -> independent frontier judge
  -> targeted validation
  -> repair memory
```

### Causal Repository Twin

`CausalRepositoryTwin` combines the persistent code index with repository-level impact relationships. It can answer which files are affected by a runtime frame and return a narrow impact slice instead of replaying the whole repository.

The V8 direction follows recent repository-agent work such as RPG/RPG-Encoder, which unifies repository structure, semantics and dependencies into a persistent representation, and ARISE, which makes fine-grained data-flow slicing a first-class localization primitive. These systems report strong gains in repository localization and scalable maintenance. 

### Runtime failure intelligence

`CausalErrorEngine` turns a traceback into a structured diagnosis containing:

- exact runtime frame
- affected dependency files
- narrow source context
- evidence-backed root-cause hypothesis
- confidence score
- prior successful repair evidence

### Repair memory

`RepairMemory` stores compact, reviewable records of failures, root causes, patches and validation outcomes. Future failures can retrieve similar successful repairs instead of rediscovering the same solution.

This follows the direction of recent memory-augmented repair research, where historical fixes and failed-to-successful refinement trajectories are reused for repository-scale repair. 

### V8 public engine

```python
from myai import AIEngine

ai = AIEngine()  # V8 by default
```

Legacy V7.1 remains available as:

```python
from myai import LegacyAIEngine
```

### Repair workflow

```python
diagnosis = ai.diagnose_failure(traceback_text)
proposal = ai.propose_repair(traceback_text)
```

V8 does not blindly overwrite production files. Patch promotion remains an explicit validation step so the system can run tests, inspect the changed impact slice and record whether the repair actually worked.

### Real model configuration

Set `MYAI_FAST_MODEL_PROVIDER`, `MYAI_BALANCED_MODEL_PROVIDER`, and `MYAI_FRONTIER_MODEL_PROVIDER` to `compatible` and provide the endpoint/model/key for each tier. Local mode remains deterministic and is useful for architecture tests.

### Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
cp .env.example .env
pytest
```

Runtime dependencies are listed in `requirements.txt`; development/CI dependencies are in `requirements-dev.txt`.

### Code-memory principle

Stable code should not be repeatedly re-read. The persistent code snapshot is freshness-aware; only changed/affected repository regions are eligible for re-indexing, and model context is limited to the relevant symbol/source slice.

Knowledge is persisted in `data/knowledge.sqlite3`; code intelligence in `data/code_index.json`; repair experience in `data/repair_memory.jsonl`.
