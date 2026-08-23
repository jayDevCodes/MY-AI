# MY-AI

MY-AI is being built as a modular AI system, upgraded through controlled versions.

## V2

V2 adds three foundations:

- Pluggable model providers through an OpenAI-compatible HTTP adapter.
- Bounded conversation memory.
- Reproducible local and CI dependency installation.

The default provider remains deterministic, so tests do not require model weights or API keys. Set `MYAI_MODEL_PROVIDER=compatible` and configure `MYAI_MODEL_BASE_URL` to connect a local or remote OpenAI-compatible inference server.

## Local setup

macOS / Linux:

```bash
./scripts/setup.sh
source .venv/bin/activate
pytest
```

Windows PowerShell:

```powershell
./scripts/setup.ps1
.\.venv\Scripts\Activate.ps1
pytest
```

Manual installation is also supported:

```bash
python -m venv .venv
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

`requirements.txt` contains runtime dependencies. `requirements-dev.txt` includes runtime dependencies plus test/lint/type-check tooling. Future versions should update these files only when a new dependency is actually introduced.

## Architecture direction

```text
core model
  -> provider adapter
  -> conversation memory
  -> retrieval / knowledge
  -> tools
  -> agents
  -> multimodal
  -> evaluation / verification
```
