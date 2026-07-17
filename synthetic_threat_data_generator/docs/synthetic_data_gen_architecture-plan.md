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

```
                 ┌─────────────────────────┐
                 │ Scenario & Persona Gen   │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Conversation Script Gen  │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Ground Truth Annotator   │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Audio Synthesis (TTS)    │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Audio Realism / Noise    │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Metadata Generator       │
                 └────────────┬────────────┘
                              ▼
                 ┌─────────────────────────┐
                 │ Packaging & Validation   │
                 └────────────┬────────────┘
                              ▼
                     Benchmark Dataset
```

### 3.1 Scenario & Persona Generator Agent
- **Input:** generation request (category distribution, sample count, config).
- **Output:** a `Scenario` (category, severity, setting/context, locale, channel-quality target) and 2+ `Persona`s (role, voice traits — pitch range/pace/accent, emotional baseline, speaking style).
- Draws from configured scenario/persona libraries (`configs/`) plus LLM-generated variation to avoid template repetition.

### 3.2 Conversation Script Generator Agent
- **Input:** `Scenario` + `Persona`s.
- **Output:** ordered list of `ConversationTurn` (speaker, text, intended emotion, turn index).
- Uses the abstracted `LLMClient` to draft realistic, natural dialogue. Threat indicators (if category ≠ `benign`) are injected at specific, controllable turns per the severity level, not spread uniformly — this makes span-level ground truth meaningful.

### 3.3 Ground Truth Annotator Agent
- **Input:** the generated script (the agent that wrote it also knows exactly where it injected signal, but this is a separate agent for independence/verification).
- **Output:** `GroundTruthLabel` set — per-turn and per-span threat labels, category, severity, speaker attribution, plus **audio-level ground truth hooks**: expected emotion per turn, expected background-event markers, expected prosody characteristics (e.g., "raised volume", "rapid pace"). These hooks are what let Project 2's *audio* agents (not just transcript agents) be evaluated against this dataset.
- Independently re-derives labels from the transcript text (not just copies injection metadata) to catch cases where the script generator's intent didn't actually land in the text — this is a QA cross-check, not a rubber stamp.

### 3.4 Audio Synthesis (TTS) Agent
- **Input:** `ConversationTurn`s + `Persona` voice traits.
- **Output:** per-turn audio clips rendered via the abstracted `TTSClient`, tagged with the target emotion/prosody from ground truth so the TTS backend (whichever is configured) is steered toward matching audio characteristics.

### 3.5 Audio Realism / Noise Augmentation Agent
- **Input:** per-turn audio clips + scenario's channel-quality/setting target.
- **Output:** a single mixed, realistic call-audio track — turns stitched with natural pacing/pauses, background noise/crosstalk overlaid per scenario, optional channel degradation (codec artifacts, dropout) simulated. Also emits the actual (post-mix) timing of each turn and any injected background events, reconciled back into ground truth (since mixing can shift timing).

### 3.6 Metadata Generator Agent
- **Output:** `CallMetadata` — call ID, start timestamp, duration, channel/telephony info, participant IDs, locale, and a pointer back to the `Scenario` used (for dataset analytics, not shown to Project 2 as a "cheat" signal — kept in a separate metadata file from ground truth).

### 3.7 Dataset Packaging & Validation Agent
- Cross-checks: does every ground truth span map to real transcript text? Does every audio timestamp fall within the actual audio duration? Are speaker labels consistent between transcript and diarization-truth? Does the category/severity distribution match the requested generation mix?
- Computes and emits a dataset **coverage report**: counts per category/severity, locale/channel-quality distribution, min/max/avg call length — so gaps in the benchmark are visible before release.
- Writes the final manifest and marks the release as valid/invalid (invalid samples are quarantined, not silently dropped).

### 3.8 Orchestrator
- LangGraph graph wiring stages 3.1–3.7 per sample; batch-runs many samples with bounded concurrency; retries transient provider failures (LLM/TTS calls) with backoff; failed samples are logged and excluded from the release rather than aborting the whole batch.

## 4. Data Schemas

Represented as Pydantic models in `src/schemas/`.

```
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

ConversationTurn
  turn_index: int
  speaker_persona_id: str
  text: str
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

DatasetManifest
  dataset_version: str
  generated_at: datetime
  samples: list[SampleRef]     # sample_id -> paths (audio, transcript, ground_truth, metadata)
  coverage_report: dict
```

The `GroundTruthLabel` schema is the most important contract in this document — Project 2's evaluation harness reads it directly to score both transcript and audio agents.

## 5. Provider Abstraction

No vendor lock-in: all model/service calls go through thin internal interfaces, backend selected via config.

```
LLMClient (interface)         # used by Scenario/Script/GroundTruth agents
  generate(prompt, **kwargs) -> str

TTSClient (interface)         # used by Audio Synthesis agent
  synthesize(text, voice_traits, target_emotion) -> AudioClip
```

Concrete implementations live under `src/providers/` (e.g., `llm_anthropic.py`, `llm_openai.py`, `tts_<vendor>.py`), selected at startup via config — no agent code references a specific vendor.

## 6. Module / Directory Structure

```
synthetic_threat_data_generator/
  src/
    agents/
      scenario_persona_agent.py
      script_generator_agent.py
      ground_truth_annotator_agent.py
      tts_agent.py
      noise_augmentation_agent.py
      metadata_agent.py
      packaging_validation_agent.py
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
    architecture-plan.md    # this file
```

## 7. Output Format

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

## 8. Milestones

- **M1 — Text-only pipeline:** Scenario/Persona → Script → Ground Truth agents working end-to-end, no audio. Validates taxonomy coverage and label quality first, cheaply.
- **M2 — Audio pipeline:** add TTS Agent + Noise Augmentation Agent; reconcile ground truth timestamps post-mix.
- **M3 — Packaging & QA:** Packaging & Validation Agent, coverage report, quarantine-on-failure.
- **M4 — Scale & versioning:** concurrent batch generation at target volume, dataset versioning/release process, regeneration diffing.

## 9. Open Questions / Future Work

- Voice diversity/accent coverage strategy — how many distinct voices per locale is "enough" for benchmark diversity.
- Multi-lingual scenario support beyond `en-US`.
- Adversarial/edge-case scenario generation (borderline cases designed to probe false-positive/false-negative boundaries in Project 2, rather than clear-cut cases).
- Whether/when to extract `LLMClient` into a shared package with Project 2's identical interface, once both are stable.
