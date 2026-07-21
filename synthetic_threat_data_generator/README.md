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

Orchestrating these agents into the full LangGraph pipeline (per architecture
doc §3) is not yet implemented -- each agent is independently callable and
tested, ready to be wired into that graph next. A rough end-to-end call order,
matching the pipeline diagram:

```python
config = configuration_agent.build_configuration(...)
scenario = scenario_generator_agent.generate_scenario(config, sample_index=0)
personas = persona_generator_agent.generate_personas(scenario)
conversation = conversation_generator_agent.generate_conversation(scenario, personas)
transcript = transcript_generator_agent.generate_transcript(conversation, personas)
ground_truth = ground_truth_generator_agent.generate_ground_truth(conversation, scenario)
# ... translate per target locale, then TTS Engine -> Audio Generator -> Dataset Exporter
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"   # or: pip install poetry && poetry install
copy .env.example .env                   # then fill in real values
```

Only `OPENAI_API_KEY` is required to exercise the LLM-based agents (Scenario,
Persona, Conversation, Transcript, Translation Generators). The TTS Engine
additionally needs `piper-tts` installed and voice models downloaded into
`PIPER_VOICES_DIR` (see `.env.example`) -- not installed by default, since it's
a heavy, optional dependency the unit tests don't need.

## Running tests

```powershell
.venv\Scripts\python -m pytest tests/ -v
```

All 11 agents are unit tested. The LLM (PydanticAI/LiteLLM) and the TTS Engine's
Piper voice loader are mocked/monkeypatched, so the suite runs in seconds with
no API keys, model downloads, or GPU required. Persona Generator, Metadata
Generator, Audio Generator, and Dataset Exporter use real Faker/pydub/file I/O
against synthetic data rather than mocks, since those dependencies are
lightweight and deterministic.
