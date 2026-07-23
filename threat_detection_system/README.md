# Threat Detection System

Enterprise multi-agent platform for audio + transcript threat detection. See
`docs/architecture/threat_detection_architecture-plan.md` for the full architecture and
`docs/database/threat_detection_database-model.md` for the database design.

## Agents

All 12 agents from the architecture plan live under `src/agents/`, one file
per agent (`_shared_detection.py` is internal plumbing shared by the four
LLM-based transcript-detection agents, not itself an agent):

| Domain | Agent | File |
|---|---|---|
| Audio | Speech-to-Text | `speech_to_text_agent.py` |
| Audio | Speaker Diarization | `speaker_diarization_agent.py` |
| Audio | Prosody Analysis | `prosody_analysis_agent.py` |
| Audio | Emotion Detection | `emotion_detection_agent.py` |
| Audio | Background Audio Detection | `background_audio_detection_agent.py` |
| Audio | Audio Correlation | `audio_correlation_agent.py` |
| Transcript | Verbal Abuse Detection | `verbal_abuse_detection_agent.py` |
| Transcript | Threat Detection | `threat_detection_agent.py` |
| Transcript | Fraud & Social Engineering | `fraud_social_engineering_agent.py` |
| Transcript | Compliance Detection | `compliance_detection_agent.py` |
| Transcript | Transcript Correlation | `transcript_correlation_agent.py` |
| Decision | Threat Correlation & Decision | `threat_correlation_decision_agent.py` |

Shared typed contracts (`AgentFinding`, `DomainScore`, `ThreatAssessment`, etc.)
are in `src/schemas.py`. Runtime configuration is in `src/config.py`. The LLM
abstraction (LiteLLM for plain-text summaries, PydanticAI for structured
detection output) is in `src/llm_client.py`.

## Orchestration

`src/orchestration/graph.py` wires all 12 agents into the LangGraph pipeline
from architecture doc §4: Speech-to-Text, Speaker Diarization, and Background
Audio Detection fan out from the raw audio; Prosody and Emotion follow once
diarization completes; the transcript-detection agents fan out once
Speech-to-Text and Speaker Diarization are merged into a diarized transcript;
both domains join at their Correlation agent, and both Correlation outputs
join at the Decision agent. Call it via:

```python
from src.orchestration.graph import run_pipeline
assessment = run_pipeline(call_id="call-1", audio_path="path/to/call.wav")
```

Building this surfaced a real LangGraph gotcha, documented in the module's
docstring: its default fan-in fires a join node as soon as *any* one incoming
edge completes, not once *all* of them have, which breaks silently when a
join's predecessors sit at different depths from the graph's start (exactly
the case for Audio Correlation and the final Decision node here). See
`src/orchestration/graph.py` and `src/orchestration/state.py` for the fix.

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"   # or: pip install poetry && poetry install
cp .env.example .env                     # then fill in real values
```

Only `OPENAI_API_KEY` is required to exercise the LLM-based agents (Verbal
Abuse, Threat, Fraud & Social Engineering, Compliance, Transcript Correlation,
Audio Correlation, Threat Correlation & Decision). The audio-ML agents
(Speech-to-Text, Speaker Diarization, Emotion Detection, Background Audio
Detection) additionally need their respective model weights, which download
automatically on first real use (Speaker Diarization also needs a
`HUGGINGFACEHUB_API_TOKEN` with the gated model's license accepted -- see
`.env.example`).

## Running tests

```bash
.venv/Scripts/python -m pytest tests/ -v
```

All 12 agents are unit tested with heavy ML backends (faster-whisper,
pyannote.audio, SpeechBrain, Silero VAD) and the LLM mocked/monkeypatched, so
the full suite runs in seconds with no API keys, model downloads, or GPU
required. Prosody Analysis is the one exception -- it uses real librosa
signal processing on synthetic audio rather than a mock, since that
dependency is lightweight and deterministic. `test_orchestration_graph.py`
covers the graph's wiring the same way -- every agent function mocked,
verifying each node runs exactly once and the right data reaches each join.
