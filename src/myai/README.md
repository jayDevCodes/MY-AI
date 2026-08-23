# V1 architecture boundaries

This package is intentionally small. Future versions plug into these boundaries without coupling the core to one model provider.

Planned modules:

- `core`: model interfaces and orchestration primitives
- `config`: environment/runtime configuration
- `memory`: short- and long-term memory
- `knowledge`: retrieval and knowledge graph integrations
- `tools`: web, code, filesystem and API tools
- `agents`: task planning and specialist agents
- `multimodal`: vision/audio/video integrations
- `evaluation`: benchmarks, regression tests and verifiers

V1 implements only configuration and health primitives so the repository has a stable, testable foundation before adding model-dependent behavior.
