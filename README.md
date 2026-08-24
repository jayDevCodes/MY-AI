# MY-AI

## V9 — Cognitive Mesh + Self-Evolving Repository Intelligence

MY-AI V9 extends V8's causal repository twin into a unified cognitive mesh for repository reasoning, targeted debugging, runtime evidence and evidence-gated self-evolution.

### V9 architecture

```text
request / failure
  -> cognitive routing
  -> unified program graph
       -> structure
       -> calls
       -> data-flow references
       -> control-flow regions
  -> causal repository twin
       -> dependency impact
       -> source slices
  -> runtime trace graph
       -> exception/state events
       -> causal links
  -> recursive multi-model workers
  -> independent judge
  -> targeted validation
  -> evolution memory
       -> strategy scores
       -> reusable lessons
       -> promotion gating
```

### Unified Program Graph

`ProgramGraph` adds a repository-level intermediate representation over the persistent code index. Instead of treating files as isolated text blobs, V9 keeps compact nodes/edges for declarations, calls, data-flow references and control-flow regions. `program_slice()` returns only the local graph neighborhood and matching source ranges needed for a task.

This follows the current repository-agent direction represented by RPG and RPG-Encoder, which use persistent repository graphs to unify structure, semantics and dependencies and evolve the representation incrementally rather than repeatedly re-reading an entire codebase. RPG-Encoder reports large maintenance-overhead reductions and strong repository localization results on SWE-bench evaluations. 

### Runtime Trace Graph

`RuntimeTraceGraph` stores compact exception/state events and causal links in `data/runtime_trace.jsonl`. The graph survives restarts, so future diagnosis can reuse runtime evidence instead of rediscovering the same sequence of events.

### Self-evolution loop

`EvolutionMemory` records task strategy, correctness, latency, token use and lessons. `EvolutionBenchmark` ranks strategies and only promotes a candidate when it is successful and materially better than the baseline. This is deliberately evidence-gated: memory can guide future strategy selection, but it cannot replace execution evidence.

Recent 2026 research on self-evolving coding agents emphasizes executable feedback, repository context and coding trajectories as the core evidence sources for safe improvement. EvoCodeBench explicitly evaluates correctness together with efficiency and improvement over repeated attempts. 

### V9 public engine

```python
from myai import AIEngine

ai = AIEngine()  # V9 by default
```

Legacy V7.1 and V8 engines remain importable for compatibility.

### Targeted repair context

```python
diagnosis = ai.diagnose_failure(traceback_text)
repair_context = ai.repair_context_v9(traceback_text)
program_slice = ai.program_slice("refresh_token")
```

The repair specialist receives the failure hypothesis, graph slice, relevant source ranges, nearby runtime events and the best historically validated strategy. Unrelated files are not required for the repair context.

### Local setup

```bash
./scripts/setup.sh
source .venv/bin/activate
cp .env.example .env
pytest
```

Runtime dependencies are listed in `requirements.txt`; development/CI dependencies are in `requirements-dev.txt`.

### Persistent state

- `data/knowledge.sqlite3` — semantic knowledge
- `data/code_index.json` — freshness-aware code index
- `data/repair_memory.jsonl` — validated repair experience
- `data/runtime_trace.jsonl` — runtime events and causal links
- `data/evolution_memory.jsonl` — strategy/evolution records

### Research grounding

V9 deliberately combines repository graphs, fine-grained program slices, executable runtime evidence and self-evolving memory rather than relying on any one technique. Current research shows memory remains task-dependent, so V9 keeps multiple evidence channels and validates promotion through actual outcomes instead of assuming that stored memory is universally reliable.
