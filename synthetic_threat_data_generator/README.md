# Synthetic Threat Data Generator

Multi-agent pipeline that produces labeled synthetic call benchmark datasets for
evaluating `threat_detection_system` (Project 2). See
`docs/architecture/synthetic_data_gen_architecture-plan.md` for the full design.

## Agents

All 11 agents from the architecture plan live under `src/agents/`, one file per
agent, in pipeline order:

| Stage | Agent | File |
|---|---|---|
| 1 | Configuration | `configuration_agent.py` |
| 2 | Scenario Generator | `scenario_generator_agent.py` |
| 3 | Persona Generator | `persona_generator_agent.py` |
| 4 | Conversation Generator | `conversation_generator_agent.py` |
| 5 | Transcript Generator | `transcript_generator_agent.py` |
| 6 | Ground Truth Generator | `ground_truth_generator_agent.py` |
| 7 | Translation Engine | `translation_engine_agent.py` |
| 8 | Metadata Generator | `metadata_generator_agent.py` |
| 9 | TTS Engine | `tts_engine_agent.py` |
| 10 | Audio Generator | `audio_generator_agent.py` |
| 11 | Dataset Exporter | `dataset_exporter_agent.py` |

Shared typed contracts (`Configuration`, `Scenario`, `Conversation`,
`GroundTruthLabel`, etc.) are in `src/schemas.py`. Runtime configuration is in
`src/config.py`. The LLM abstraction (LiteLLM for plain-text generation,
PydanticAI for structured generation) is in `src/llm_client.py`.

## Orchestration

`src/orchestration/graph.py` wires 10 of the 11 agents (everything except
Configuration, which validates a whole batch once, not per sample) into a
LangGraph pipeline that produces one sample: Scenario -> Persona ->
Conversation run as a strict chain, then Ground Truth -> Transcript ->
Translation -> TTS -> Audio Generator run sequentially while Metadata
Generator runs independently, and Dataset Exporter joins the two.

From a terminal, use `generate.py` (see `generate.py --help`, or
`docs/setup/synthetic_data_gen_running-the-pipeline.md` for more examples):

```powershell
.venv\Scripts\python.exe generate.py --category threat_of_violence --count 1
```

Or call `generate_dataset` directly:

```python
from src.orchestration.graph import generate_dataset

manifest = generate_dataset(
    category_distribution={"benign": 0.5, "verbal_abuse": 0.5},
    sample_count=10,
    locales=["en-US"],
    dataset_version="v1",
    output_dir="datasets/v1",
)
```

Ground Truth Generator runs sequentially rather than in true parallel with
the transcript branch, which the architecture doc's diagram depicts as
parallel -- a deliberate LangGraph-specific adaptation, not a change in what
Ground Truth Generator depends on. See `src/orchestration/graph.py`'s module
docstring for the full "why," including a real LangGraph gotcha it ran into:
`defer=True` (the usual fix for a join whose predecessors sit at unequal
depth) is documented as deferring a node "until the run is about to end" --
built for one true terminal barrier, not a chain of intermediate joins. Using
it on two chained joins caused the later one to fire before the earlier one
had even run.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # or: pip install poetry && poetry install
copy .env.example .env                   # then fill in real values
```

Only `OPENAI_API_KEY` is required to exercise the LLM-based agents (Scenario,
Persona, Conversation, Transcript, Translation Generators). The TTS Engine
additionally needs `piper-tts` installed and voice models downloaded into
`PIPER_VOICES_DIR` (see `.env.example`) -- not installed by default, since
it's a heavy, optional dependency the unit tests don't need:

```powershell
.venv\Scripts\pip install piper-tts
.venv\Scripts\python -m piper.download_voices --download-dir configs\voices en_US-amy-medium en_US-ryan-high en_US-hfc_male-medium en_US-hfc_female-medium
```

(All four -- not just one -- since `tts_engine_agent.py` assigns each
speaker in a call a gender-appropriate voice from these two male/two female
pools, alternating gender across speakers so two people on the same call
never sound alike; a run can hit any of the four. Downloaded files land in
`configs/voices/`, which is gitignored.)

## Running tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

All 11 agents are unit tested. The LLM (PydanticAI/LiteLLM) and the TTS Engine's
Piper voice loader are mocked/monkeypatched, so the suite runs in seconds with
no API keys, model downloads, or GPU required. Persona Generator, Metadata
Generator, Audio Generator, and Dataset Exporter use real Faker/pydub/file I/O
against synthetic data rather than mocks, since those dependencies are
lightweight and deterministic. `test_orchestration_graph.py` covers the
graph's wiring the same way -- every agent function mocked, verifying each
node runs exactly once and `generate_dataset` produces one sample per
requested count.
