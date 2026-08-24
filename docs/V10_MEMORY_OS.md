# V10 Memory OS Lifecycle

MY-AI retains raw episodic memories as evidence and applies a deterministic lifecycle policy around them.

- Repeated, consistently high-confidence episodes can be promoted to semantic/procedural memory.
- Source episodes are never rewritten or deleted by consolidation.
- Retention uses importance, confidence, age decay, and memory type.
- Promotion is intentionally conservative to avoid compounding LLM-generated memory errors.

The engine runs consolidation periodically rather than after every interaction, while the bounded CognitiveContext controls what reaches model prompts.