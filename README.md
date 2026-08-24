# MY-AI

## V10 — Cognitive Mesh + Self-Healing Runtime

MY-AI V10 extends V9.1 with a bounded self-healing runtime on top of the cognitive mesh. The system keeps the repository/program/causal graphs, persistent runtime traces, recursive multi-model workers, capability ledger, and evidence-gated evolution while adding failure signatures, bounded reproduction, repair episodes, controlled fault injection, and stable-code health metadata.

### V10 architecture

```text
request / failure
  -> cognitive routing
  -> unified program graph
       -> structure / calls / data-flow / control-flow
  -> causal repository twin
       -> dependency impact / source slices
  -> runtime trace graph
       -> state / exception / causal links
  -> failure signature memory
  -> bounded reproduction
  -> recursive repair workers + independent judge
  -> sandbox / validation boundary
  -> fault-injection self-tests
  -> stable-code health map
  -> verified strategy / capability ledger
```

### Self-healing principles

1. Detect before repairing.
2. Reproduce before trusting a hypothesis whenever possible.
3. Use the smallest causally relevant context; preserve verified code.
4. Keep retries and repair attempts bounded.
5. Never auto-promote code from the repair runtime; promotion remains validation-gated.
6. Convert verified repair experience into reusable memory and regression evidence.

### Runtime controls

- `MYAI_SELF_HEALING_ENABLED=true`
- `MYAI_SELF_HEALING_MAX_REPAIR_ATTEMPTS=2`
- `MYAI_SELF_HEALING_REPRODUCTION_TIMEOUT_SECONDS=30`
- `MYAI_FAILURE_SIGNATURE_PATH=data/failure_signatures.jsonl`
- `MYAI_REPAIR_EPISODE_PATH=data/repair_episodes.jsonl`
- `MYAI_CODE_HEALTH_PATH=data/code_health.json`
