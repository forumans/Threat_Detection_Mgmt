# Synthetic Threat Data Generator — Architecture & Implementation Plan

## 1. Overview & Goals

**Purpose.** This project is a multi-agent framework that produces synthetic, fully-labeled call datasets — audio, transcripts, ground truth, and metadata — for training and, primarily, for **evaluating** Project 2 (`threat_detection_system`). Every dataset it emits is a *benchmark release*: a versioned bundle of samples with known-correct labels that Project 2's evaluation harness can score itself against.

**Goals**
- Generate diverse, realistic call scenarios spanning the full threat taxonomy Project 2 is built to detect, plus benign negative controls.
- Produce paired audio + transcript + ground truth + metadata for each sample, internally consistent (timestamps align, speaker labels align, labels are traceable to specific transcript spans/audio segments).
- Support scenario coverage that exercises both transcript-level signals (abusive language, threats, fraud, compliance violations) and audio-level signals (emotion, prosody, background events) since Project 2 has agents for both.
- Package output as versioned, self-describing benchmark releases that can be consumed by an automated evaluation harness.

**Non-goals**
- This is not a production call-handling or telephony system — it produces offline synthetic data only.
- It does not itself score or evaluate Project 2; it only produces the fixtures. Evaluation logic lives in Project 2 (see its `evaluation/` module).
- No real customer data is ever used or required — all content is generated/synthesized.

## 2. Threat Taxonomy

The taxonomy is the shared contract with Project 2 — every category here must map to a detection agent there, and vice versa.

| Category | Description | Maps to (Project 2 agent) |
|---|---|---|
| `verbal_abuse` | Insults, harassment, degrading language directed at a party on the call | Verbal Abuse Detection Agent |
| `threat_of_violence` | Explicit or implicit threats of physical harm | Threat Detection Agent |
| `fraud_social_engineering` | Scam attempts, impersonation, manipulation to extract money/credentials/info | Fraud & Social Engineering Agent |
| `compliance_violation` | Required disclosures missing, PII/PCI exposure, policy/script deviations | Compliance Detection Agent |
| `benign` | Normal, non-threatening call — negative control | (all agents should emit low/no findings) |

Each category (except `benign`) carries a **severity**: `low`, `medium`, `high`, `critical`. Severity drives how strongly the threat indicator is expressed in dialogue and audio (e.g., a `critical threat_of_violence` sample has unambiguous, escalating language and matching vocal stress; a `low` one is a borderline/ambiguous edge case — useful for testing precision, not just recall).

Scenarios should also vary along orthogonal axes independent of category: language/locale, channel quality (clean vs. degraded audio), number of speakers (2-party vs. multi-party/conference), and call length.

## 3. Pipeline & Agent Design

Pipeline shape (per the SRS diagram): `Dataset Generator → Synthetic Conversations → Synthetic Audio Files → Benchmark Dataset`. Implemented as a LangGraph graph with mostly-sequential stages per sample, fanned out across many samples for batch generation.

A few architectural recommendations
- Separate generation stages into plugins. Define interfaces for scenarios, personas, conversations, translations, TTS, and metadata so each can be replaced independently.
- Generate structured data first, audio second. The transcript and metadata should be the canonical source; audio is a rendered artifact. This makes regeneration and multilingual expansion much easier.
- Version everything that affects dataset generation. Persist prompt versions, model versions, configuration, and random seeds so every dataset can be reproduced exactly.
- Keep Project 1 independent of Project 2. The generator should produce benchmark artifacts without any knowledge of how the threat detection platform works. The only coupling should be the shared data contracts (transcript schema, metadata schema, and ground-truth schema). This separation will make both systems easier to evolve and test independently.
- Generate the ground truth immediately after the conversation is created, before translation or audio rendering. This ensures the labels are derived from the canonical conversation and are language-independent.
- Treat the conversation object as the single source of truth. Everything else—transcripts, translations, audio, metadata, and benchmark packages—should be generated from that object rather than from one another. That keeps every artifact synchronized and makes it easy to regenerate audio or add new languages later without recreating the entire dataset.
- This pipeline is scalable, reproducible, and well suited to producing high-quality multilingual benchmark datasets for the downstream threat detection platform.

```mermaid
                    Configuration
                          │
                          ▼
                  Scenario Generator
                          │
                          ▼
                  Persona Generator
                          │
                          ▼
                 Conversation Generator
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
  Transcript Generator             Ground Truth Generator
          │                               │
          ▼                               ▼
   Translation Engine              Metadata Generator
          │                               │
          ▼                               │
      TTS Engine                          │
          │                               │
          ▼                               │
   Audio Generator                        │
          │                               │
          └───────────────┬───────────────┘
                           ▼
                   Dataset Exporter
```

Each box above is one agent, detailed below in pipeline order. Every agent accepts and returns a typed Pydantic model (§5) and is independently testable, per the "typed contracts everywhere" / "keep agents stateless" principles noted above.

### 3.1 Configuration Agent
- **Input:** generation request (category distribution, sample count, config).
- **Output:** a `Configuration` object containing the generation request.
- This agent is responsible for validating the generation request and ensuring that the configuration is consistent and complete.

### 3.2 Scenario Generator
- **Input:** the `Configuration` object (requested category distribution, sample count, taxonomy).
- **Output:** a `Scenario` (category, severity, setting, locale, channel_quality, num_speakers — see §5).
- Draws from configured scenario templates (`configs/scenarios/`) plus LLM-generated variation (via the `LLMClient` / PydanticAI agent framework, §6) to avoid repetitive, templated scenarios while still hitting the category/severity mix `Configuration` requested.

### 3.3 Persona Generator
- **Input:** the `Scenario` (locale, setting, num_speakers).
- **Output:** one `Persona` per speaker (role, voice_traits, emotional_baseline — see §5).
- Draws from a configured persona library (`configs/personas/`) plus LLM-generated variation, matching each persona's voice traits and emotional baseline to the scenario's setting and category (e.g. a collections-call agent persona reads differently than a caller persona).

### 3.4 Conversation Generator
- **Input:** the `Scenario` + its `Persona`s.
- **Output:** the canonical `Conversation` object — an ordered, language-independent turn plan (per-turn speaker, intended content/intent, intended emotion, and where threat indicators are injected).
- This is the single source of truth the architectural principles above call for: every downstream stage (Transcript, Ground Truth, Metadata, and ultimately Audio) derives from this one object rather than from each other, so they can never drift out of sync. Threat indicators (for non-`benign` scenarios) are injected at specific, controllable turns per the scenario's severity rather than spread uniformly — this is what makes span-level ground truth meaningful.

### 3.5 Transcript Generator
- **Input:** the canonical `Conversation` object.
- **Output:** a `ConversationTurn` list (transcript.json) — the actual verbatim dialogue text per turn, in the scenario's source locale.
- Expands each turn's planned intent/content into natural, verbatim dialogue text. This is a rendering step, not a planning step — it decides *how* something is phrased, not *what* happens in the call (that's Conversation Generator's job).

### 3.6 Ground Truth Generator
- **Input:** the canonical `Conversation` object -- deliberately *not* the rendered transcript, per the "generate ground truth before translation or audio rendering" principle above.
- **Output:** the `GroundTruthLabel` list — per-turn/per-span threat labels, category, severity, speaker attribution, plus the audio-level hooks (`expected_emotion`, `expected_background_event`, `expected_prosody_notes`) that make Project 2's audio agents evaluable too.
- Labels come from the conversation's planned intent/injection points, not from re-parsing rendered text. That keeps ground truth language-independent (correct for every locale Translation Engine later produces) and avoids the circularity of a later stage "grading its own homework."

### 3.7 Translation Engine
- **Input:** the source-locale `ConversationTurn` list + the target locale(s) from `Scenario`/`Configuration`.
- **Output:** a target-locale `ConversationTurn` list.
- Translates the transcript into each target locale via an LLM and/or a dedicated MT model (NLLB/MarianMT, see §4). A pass-through/no-op when the scenario's locale matches the canonical generation locale -- translation is conditional, not mandatory for every sample.

### 3.8 Metadata Generator
- **Input:** the canonical `Conversation` object + `Scenario`.
- **Output:** `CallMetadata` (call_id, timestamps, duration, channel/telephony info, participant IDs, locale — see §5).
- Generates call-level metadata independent of the rendered transcript/audio, keeping a pointer back to the `Scenario` used for dataset analytics -- stored in a file separate from ground truth so it's never visible to Project 2 as an evaluation "cheat" signal.

### 3.9 TTS Engine
- **Input:** the (possibly translated) `ConversationTurn` list + `Persona` voice traits.
- **Output:** per-turn synthesized audio clips.
- Renders each turn to speech via the abstracted `TTSClient` (§6, backed by Piper/Coqui TTS/XTTS v2, see §4), steered toward each turn's intended emotion so the rendered audio's prosody/emotion matches what Ground Truth Generator already recorded as `expected_emotion` / `expected_prosody_notes`.

### 3.10 Audio Generator
- **Input:** per-turn TTS audio clips + the `Scenario`'s channel-quality/setting target.
- **Output:** the final mixed call-audio track (`audio.wav`).
- Stitches turns together with natural pacing/pauses, overlays background noise/crosstalk per the scenario's setting, and applies optional channel degradation (codec artifacts, dropout). Reconciles each turn's actual post-mix timing -- and any injected background events -- back into `GroundTruthLabel.audio_start_ms` / `audio_end_ms`, since mixing can shift timing from the original per-clip estimates.

### 3.11 Dataset Exporter
- **Input:** everything produced for one sample -- audio, transcript, ground truth, metadata.
- **Output:** the packaged, validated sample and, across a batch, the versioned benchmark release (`manifest.jsonl` + `coverage_report.json`, see §8).
- Cross-checks consistency across all four artifacts (ground-truth spans map to real transcript text, audio timestamps fall within actual audio duration, speaker labels agree across files), computes the dataset coverage report, and writes the final manifest. Invalid samples are quarantined rather than silently dropped or allowed into the release.

## 4. Recommended Tech Stack

| Category | Recommended Technologies |
|----------|-------------------------|
| Language | Python 3.12+ |
| Backend | FastAPI, Uvicorn |
| Validation | Pydantic v2 |
| ORM | SQLAlchemy, Alembic |
| Orchestration | LangGraph |
| LLM Abstraction | LiteLLM |
| Agent Framework | PydanticAI |
| Prompt Templates | Jinja2 |
| Translation | LLM + NLLB/MarianMT (optional) |
| TTS | Piper, Coqui TTS, XTTS v2 |
| Audio Processing | FFmpeg, pydub, librosa, soundfile |
| Data Generation | Faker, Mimesis |
| Database | PostgreSQL |
| Storage | MinIO |
| Config | pydantic-settings, PyYAML |
| Testing | pytest, hypothesis |
| Docs | MkDocs Material, Mermaid |
| Packaging | Poetry |
| Quality | Ruff, Black, MyPy |
| CI/CD | GitHub Actions |
| Containers | Docker, Docker Compose |

## 5. Data Schemas

Represented as Pydantic models in `src/schemas/`. Grouped as **Domain Models** — the
persisted entities that flow through the pipeline in §3 and end up in a benchmark
release (§8). (See "Other model tiers considered" at the end of this section for why
this stays a single, flat tier rather than the Domain/Generator/Agent/API split
considered during drafting.)

```
Configuration
  config_id: str
  category_distribution: dict[str, float]   # requested mix of taxonomy categories, incl. "benign"
  sample_count: int
  locales: list[str]
  seed: int | None                # for reproducible generation, per the versioning principle above

Scenario
  scenario_id: str
  category: Literal["verbal_abuse","threat_of_violence","fraud_social_engineering","compliance_violation","benign"]
  severity: Literal["low","medium","high","critical"] | None   # None for benign
  setting: str            # e.g. "customer support call", "collections call"
  locale: str             # e.g. "en-US"
  channel_quality: Literal["clean","degraded"]
  num_speakers: int

Persona
  persona_id: str
  role: str                # e.g. "caller", "agent"
  voice_traits: dict       # pitch_range, pace, accent, timbre
  emotional_baseline: str

Conversation
  conversation_id: str
  scenario_id: str
  turns: list[ConversationTurnPlan]   # planned intent/content per turn, pre-rendering
  injected_threat_turn_indices: list[int]   # which turns carry the scenario's threat indicators

ConversationTurnPlan
  turn_index: int
  speaker_persona_id: str
  intended_content: str    # what this turn should communicate, not yet phrased as dialogue
  intended_emotion: str

ConversationTurn
  turn_index: int
  speaker_persona_id: str
  text: str                # the rendered, verbatim dialogue -- Transcript Generator's output
  intended_emotion: str

GroundTruthLabel
  sample_id: str
  turn_index: int
  span: tuple[int, int] | None   # char offsets within turn text, None if turn-level only
  category: str
  severity: str | None
  speaker_persona_id: str
  audio_start_ms: int | None     # populated after audio mixing
  audio_end_ms: int | None
  expected_emotion: str | None
  expected_background_event: str | None
  expected_prosody_notes: str | None

CallMetadata
  call_id: str
  scenario_id: str
  start_timestamp: datetime
  duration_ms: int
  channel_info: dict
  participant_ids: list[str]
  locale: str

AudioFile
  sample_id: str
  path: str                # audio.wav location within the sample folder, see §8
  duration_ms: int
  sample_rate_hz: int
  channel_quality: Literal["clean","degraded"]   # echoes Scenario.channel_quality, post-mix

DatasetManifest
  dataset_version: str
  generated_at: datetime
  samples: list[SampleRef]     # sample_id -> paths (audio, transcript, ground_truth, metadata)
  coverage_report: dict
```

The `GroundTruthLabel` schema is the most important contract in this document — Project 2's evaluation harness reads it directly to score both transcript and audio agents.

**Other model tiers considered.** An earlier draft of this section proposed splitting
models into four tiers: Domain, Generator (request), Agent (input/output), and API
models. That's folded back into the single tier above, because:
- **Generator-request wrappers were redundant.** Every stage in §3 already has an
  explicit `Input:`/`Output:` pair, and each input is just the prior stage's domain
  object (Persona Generator's input is a `Scenario`, not a separate
  `PersonaGenerationRequest`). A wrapper type with no fields beyond what the domain
  object already has doesn't earn its own schema.
- **Agent input/output models belong to Project 2, not here.** `AudioAgentInput` /
  `TranscriptAgentInput` describe what Project 2's detection agents consume — but this
  project's own Non-goals (§1) say it should have no knowledge of how the threat
  detection platform works. Project 2's `docs/database/threat_detection_database-model.md`
  already defines that contract from its side; duplicating it here risks the two
  drifting apart.
- **API request models are premature.** `GenerateDatasetRequest` / `AnalyzeAudioRequest`
  / `BenchmarkRequest` imply a REST API surface (FastAPI is in §4's tech stack as a
  future option), but no endpoint design exists yet — no milestone in §9 covers one.
  These are noted in §10 (Open Questions) as future work instead of speculatively
  schema'd now.

## 6. Provider Abstraction

No vendor lock-in: all model/service calls go through thin internal interfaces, backend selected via config.

```
LLMClient (interface)         # used by Scenario Generator, Persona Generator,
                               # Conversation Generator, Transcript Generator,
                               # Ground Truth Generator, and (optionally) Translation Engine
  generate(prompt, **kwargs) -> str

TTSClient (interface)         # used by TTS Engine
  synthesize(text, voice_traits, target_emotion) -> AudioClip
```

Concrete implementations live under `src/providers/` (e.g., `llm_anthropic.py`, `llm_openai.py`, `tts_<vendor>.py`), selected at startup via config — no agent code references a specific vendor.

## 7. Module / Directory Structure

```
synthetic_threat_data_generator/
  src/
    agents/
      configuration_agent.py
      scenario_generator_agent.py
      persona_generator_agent.py
      conversation_generator_agent.py
      transcript_generator_agent.py
      ground_truth_generator_agent.py
      translation_engine_agent.py
      metadata_generator_agent.py
      tts_engine_agent.py
      audio_generator_agent.py
      dataset_exporter_agent.py
    orchestration/
      graph.py            # LangGraph graph definition
      state.py            # per-sample generation state
    providers/
      llm_client.py        # interface
      llm_<vendor>.py
      tts_client.py         # interface
      tts_<vendor>.py
    schemas/
      scenario.py
      persona.py
      transcript.py
      ground_truth.py
      metadata.py
      manifest.py
    pipeline/
      batch_runner.py      # CLI entrypoint, concurrency/retry control
    storage/
      dataset_writer.py
      object_store_client.py   # S3-compatible (MinIO/S3/Azure Blob/GCS via config)
  configs/
    scenarios/              # scenario templates per category
    personas/                # persona library
    generation.yaml          # batch mix, concurrency, provider selection
  tests/
  docs/
    architecture/
      synthetic_data_gen_architecture-plan.md    # this file
```

## 8. Output Format

A benchmark release is a versioned folder (or object-store prefix): `datasets/v{N}/`:

```
datasets/v1/
  manifest.jsonl              # one line per sample: paths + summary
  coverage_report.json
  samples/
    <sample_id>/
      audio.wav
      transcript.json         # ConversationTurn list
      ground_truth.json       # GroundTruthLabel list
      metadata.json           # CallMetadata
```

`manifest.jsonl` is the single entrypoint Project 2's evaluation harness reads to enumerate a release.

## 9. Milestones

- **M1 — Text-only pipeline:** Configuration → Scenario Generator → Persona Generator → Conversation Generator → Transcript Generator → Ground Truth Generator working end-to-end, no audio, no translation. Validates taxonomy coverage and label quality first, cheaply.
- **M2 — Multilingual text:** add Translation Engine; verify ground truth stays correct across target locales (it's derived from the pre-translation `Conversation` object, so it should need no changes).
- **M3 — Audio pipeline:** add TTS Engine + Audio Generator + Metadata Generator; reconcile ground truth timestamps post-mix.
- **M4 — Packaging & QA:** Dataset Exporter, coverage report, quarantine-on-failure.
- **M5 — Scale & versioning:** concurrent batch generation at target volume, dataset versioning/release process, regeneration diffing.

## 10. Open Questions / Future Work

- Voice diversity/accent coverage strategy — how many distinct voices per locale is "enough" for benchmark diversity.
- Multi-lingual scenario support beyond `en-US`.
- Adversarial/edge-case scenario generation (borderline cases designed to probe false-positive/false-negative boundaries in Project 2, rather than clear-cut cases).
- Whether/when to extract `LLMClient` into a shared package with Project 2's identical interface, once both are stable.
- A REST API surface (FastAPI, per §4) for triggering generation and querying releases -- e.g. `GenerateDatasetRequest`, `AnalyzeAudioRequest`, `BenchmarkRequest` -- once there's an actual endpoint design to schema against; no milestone in §9 covers this yet.
└── pyproject.toml