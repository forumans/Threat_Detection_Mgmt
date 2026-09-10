# Threat Detection Mgmt

A multi-agent threat-detection platform for call audio. This repository holds
two related, independently-runnable projects that together form an evaluable
system:

| Project | Role |
|---|---|
| [`synthetic_threat_data_generator/`](synthetic_threat_data_generator/) | Produces versioned, fully-labeled synthetic call datasets (audio + transcript + ground truth + metadata) used to benchmark Project 2 |
| [`threat_detection_system/`](threat_detection_system/) | Multi-agent platform that analyzes call audio + transcript to detect verbal abuse, threats, fraud/social engineering, and compliance violations, and correlates them into a risk score |

The two projects are deliberately decoupled — Project 1 has no knowledge of how
Project 2 works — and share exactly one contract: the `GroundTruthLabel` schema
that Project 2's evaluation harness reads to score itself against Project 1's
releases (see [§12 of the Threat Detection System's architecture plan](threat_detection_system/docs/architecture/threat_detection_architecture-plan.md#12-evaluation-loop-integration-with-project-1)).

**Threat Detection Mgmt is a multi-agent system end to end.** Across both
projects, 23 single-responsibility agents (11 in Project 1, 12 in Project 2) —
each a stateless function that accepts a typed Pydantic model and returns one —
are orchestrated as [LangGraph](https://github.com/langchain-ai/langgraph)
graphs rather than a monolithic pipeline: agents fan out to run in parallel
wherever their inputs allow it, join at correlation/aggregation points, and
never call each other directly. That design is what makes each agent
independently testable, replaceable, and (in Project 2) independently scored
against Project 1's ground truth.

## Architecture

### Synthetic Threat Data Generator (Project 1)

A LangGraph pipeline of 11 agents that turns a generation request into a batch
of labeled call samples. Each sample carries audio, a transcript, ground-truth
labels, and metadata that are all derived from one canonical `Conversation`
object, so they can never drift out of sync with each other:

```
Configuration → Scenario Generator → Persona Generator → Conversation Generator
                                                                  │
                                        ┌─────────────────────────┴─────────────────────────┐
                                        ▼                                                     ▼
                                Transcript Generator                                  Ground Truth Generator
                                        │                                                     │
                                        ▼                                                     ▼
                                Translation Engine                                    Metadata Generator
                                        │                                                     │
                                        ▼                                                     │
                                    TTS Engine                                                │
                                        │                                                     │
                                        ▼                                                     │
                                 Audio Generator                                              │
                                        └─────────────────────────┬─────────────────────────┘
                                                                   ▼
                                                           Dataset Exporter
```

- **Threat taxonomy:** `verbal_abuse`, `threat_of_violence`, `fraud_social_engineering`,
  `compliance_violation`, plus `benign` as a negative control — each category maps
  1:1 to a detection agent in Project 2. Non-benign categories carry a severity
  (`low`/`medium`/`high`/`critical`).
- **Ground truth first:** the Ground Truth Generator labels the pre-translation,
  pre-audio `Conversation` object directly (not the rendered transcript), keeping
  labels language-independent and avoiding a later stage "grading its own homework."
- **Provider abstraction:** LLM calls go through an `LLMClient` interface, speech
  synthesis through a `TTSClient` interface — concrete vendors are swapped via config.
- **Output:** a versioned benchmark release under `datasets/v{N}/`, with one
  `manifest.jsonl` line per sample pointing at that sample's `audio.wav`,
  `transcript.json`, `ground_truth.json`, and `metadata.json`.

Full design: [`docs/architecture/synthetic_data_gen_architecture-plan.md`](synthetic_threat_data_generator/docs/architecture/synthetic_data_gen_architecture-plan.md) · sequence diagram: [`docs/diagrams/`](synthetic_threat_data_generator/docs/diagrams/).

### Threat Detection System (Project 2)

A LangGraph pipeline of 12 agents across 3 domains that analyzes a call's audio
and transcript in parallel, then correlates both domains into a single,
explainable risk assessment:

```
                      Call Audio + (optional) Reference Transcript
                                       │
                  ┌────────────────────┴────────────────────┐
                  ▼                                          ▼
         Audio Intelligence Domain                Transcript Intelligence Domain
   (STT, Diarization, Prosody,                    (Verbal Abuse, Threat, Fraud &
    Emotion, Background Audio)                     Social Eng., Compliance)
                  │                                          │
                  ▼                                          ▼
         Audio Correlation Agent                  Transcript Correlation Agent
                  └────────────────────┬────────────────────┘
                                       ▼
                     Threat Correlation & Decision Agent
                                       ▼
                     Risk Score + Findings + Alert Decision
                                       │
                ┌──────────────────────┼──────────────────────┐
                ▼                      ▼                      ▼
         Supervisor Alerts        Reporting             REST API → Dashboard
```

- **Audio domain:** Speech-to-Text and Speaker Diarization run in parallel off the
  raw audio and merge into a diarized transcript; Prosody, Emotion, and Background
  Audio Detection run in parallel directly off the raw audio; all five join at
  **Audio Correlation**.
- **Transcript domain:** once a transcript is available, Verbal Abuse, Threat,
  Fraud & Social Engineering, and Compliance run in parallel, joining at
  **Transcript Correlation**.
- **Decision:** both correlation outputs join at the **Threat Correlation &
  Decision Agent**, which combines a weighted `audio_domain_score` and
  `transcript_domain_score` into an overall risk score — corroboration across
  domains (e.g. a threatening transcript span backed by high-arousal audio) scores
  higher than either signal alone.
- **Delivery:** alert decisions go out through a `NotificationChannel` interface
  (webhook-based; Slack/email/PagerDuty as adapters); a FastAPI service exposes
  `POST /calls`, `GET /calls`, `GET /calls/{call_id}`, `GET /alerts`, and
  `POST /evaluation/run` for the dashboard and for CI/regression evaluation
  against Project 1 releases.
- **Storage:** Postgres for call records/findings/assessments/alerts; S3-compatible
  object storage (MinIO locally) for audio, both behind a `storage/` abstraction.
- **Phasing:** Phase 1 is batch (post-call) processing; streaming/live-call
  analysis is an explicit, deferred Phase 2 extension.

Full design: [`docs/architecture/threat_detection_architecture-plan.md`](threat_detection_system/docs/architecture/threat_detection_architecture-plan.md) · database model: [`docs/database/threat_detection_database-model.md`](threat_detection_system/docs/database/threat_detection_database-model.md) · sequence diagram: [`docs/diagrams/`](threat_detection_system/docs/diagrams/).

## Tools & Technologies

Every agent's external calls — LLM, speech-to-text, text-to-speech, alert
delivery — go through a small provider interface (`LLMClient`, `STTClient`,
`TTSClient`, `NotificationChannel`) rather than a hardcoded vendor SDK, so the
concrete tool behind each is swapped via config. The tools currently wired in:

| Category | Synthetic Threat Data Generator (Project 1) | Threat Detection System (Project 2) |
|---|---|---|
| Agent orchestration | [LangGraph](https://github.com/langchain-ai/langgraph) | [LangGraph](https://github.com/langchain-ai/langgraph) |
| Agent development | [PydanticAI](https://ai.pydantic.dev/) | [PydanticAI](https://ai.pydantic.dev/) |
| LLM abstraction | [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI by default) | [LiteLLM](https://github.com/BerriAI/litellm) (OpenAI by default) |
| Speech-to-text | — | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| Speaker diarization | — | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| Emotion recognition | — | [SpeechBrain](https://speechbrain.github.io/) |
| Voice activity detection | — | [Silero VAD](https://github.com/snakers4/silero-vad) |
| Text-to-speech | [Piper TTS](https://github.com/rhasspy/piper) | — |
| Audio processing | librosa, pydub, soundfile, FFmpeg | librosa, torchaudio, pydub, soundfile |
| Synthetic data generation | Faker, Mimesis | — |
| Backend / API | FastAPI, Uvicorn, Pydantic, SQLAlchemy | FastAPI, Uvicorn, Pydantic, SQLAlchemy |
| Database / storage | PostgreSQL, S3-compatible object storage (MinIO) | PostgreSQL, S3-compatible object storage (MinIO) |
| Config | pydantic-settings, PyYAML, python-dotenv | pydantic-settings, PyYAML, python-dotenv |
| Testing | pytest, hypothesis | pytest, hypothesis, pytest-mock |
| Packaging | Poetry | Poetry |

See each project's `pyproject.toml` for exact dependency versions, and
[`docs/architecture/tech_stack_guidelines.md`](threat_detection_system/docs/architecture/tech_stack_guidelines.md)
for the broader recommended stack (dashboard, caching, monitoring, etc.) that
Phase 1 hasn't needed yet.

## Getting started

Each project is set up and run independently — see its own README:

- [`synthetic_threat_data_generator/README.md`](synthetic_threat_data_generator/README.md)
- [`threat_detection_system/README.md`](threat_detection_system/README.md)
