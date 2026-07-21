# Local Setup & Running the Project

How to get `threat_detection_system` running on your machine today. See also
`README.md` (agent catalog + file map) and `threat_detection_architecture-plan.md`
(overall design).

## Status check first

There is no application entrypoint yet — no API server, no CLI, no LangGraph
orchestration wiring the 12 agents together into the pipeline described in the
architecture plan. What exists today is: all 12 agents, independently callable
and unit tested, plus a provisioned (but not yet agent-connected) database.
"Running the project" currently means running the test suite and/or invoking
an agent directly, per below.

## 1. Prerequisites

- Python 3.12+
- An OpenAI API key (for the LLM-based agents; see step 3)
- PostgreSQL, if you want the database up too (optional for running agents)

## 2. Set up the environment

A venv already exists at `.venv/` if you're continuing on a machine where it
was created. To (re)create it from scratch:

```powershell
cd threat_detection_system
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]"
```

(`pyproject.toml` is Poetry-formatted; `poetry install` works instead once
Poetry is installed. The `pip install -e ".[dev]"` path above is what was
actually used to get the current `.venv` running.)

To activate it in PowerShell for the rest of these steps:

```powershell
.venv\Scripts\Activate.ps1
```

Note: only the lightweight dependencies (pydantic, pydantic-ai, litellm,
openai, librosa, pytest, ...) are installed by default. The heavy audio-ML
libraries (faster-whisper, pyannote.audio, speechbrain, silero-vad) are
declared in `pyproject.toml` but not installed yet — install them individually
if you need to run the Speech-to-Text, Speaker Diarization, Emotion Detection,
or Background Audio Detection agents against real audio (see step 5).

## 3. Configure environment variables

```powershell
copy .env.example .env
```

Then fill in `.env`. At minimum, set `OPENAI_API_KEY` — that's all you need
to run the six LLM-based agents (Verbal Abuse, Threat, Fraud & Social
Engineering, Compliance, Audio Correlation, Transcript Correlation, Threat
Correlation & Decision). `.env` is gitignored; never commit real keys.

The remaining variables (`HUGGINGFACEHUB_API_TOKEN`, `WHISPER_MODEL_SIZE`,
etc.) are only needed for the audio-ML agents — see `.env.example` for what
each one is for and where to get it.

## 4. Run the test suite

```powershell
python -m pytest tests/ -v
```

All 12 agents are covered (33 tests total). Heavy ML backends and the LLM are
mocked/monkeypatched throughout, so this requires no API keys, no model
downloads, and runs in a few seconds. The one exception is Prosody Analysis,
which uses real librosa signal processing on synthetic audio rather than a
mock, since that dependency is lightweight and deterministic.

## 5. Run an agent directly

There's no CLI yet, so invoke an agent as a Python function. Example (LLM-based
agent, needs `OPENAI_API_KEY` set in `.env`):

```python
from src.agents.verbal_abuse_detection_agent import detect_verbal_abuse
from src.schemas import TranscriptTurn

turns = [
    TranscriptTurn(turn_index=0, speaker_label="caller", start_ms=0, end_ms=2000,
                    text="You are a worthless idiot."),
]
print(detect_verbal_abuse(turns))  # makes a real call to OpenAI
```

Audio agents take an `AudioInput(call_id=..., audio_path=...)` pointing at a
real audio file, and require their specific library installed first (e.g.
`pip install faster-whisper` for Speech-to-Text). Speaker Diarization
additionally requires accepting the gated model's license on Hugging Face and
setting `HUGGINGFACEHUB_API_TOKEN` (see `.env.example`).

## 6. Database (optional, not yet wired to the agents)

The schema is provisioned separately from the agents (nothing in `src/`
reads/writes it yet):

```powershell
psql -h localhost -U <superuser> -d postgres -f db_scripts\00_create_database.sql
psql -h localhost -U <superuser> -d threat_detection -f db_scripts\schema.sql
```

`DATABASE_URL` in `.env` should already point at this database.

## What's next

Wiring the 12 agents into the LangGraph pipeline from the architecture plan
(§4) and adding a FastAPI entrypoint (§8) is what turns this from "a set of
tested agents" into an actual runnable service — that work hasn't started yet.
