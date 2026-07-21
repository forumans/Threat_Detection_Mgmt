# Threat Detection System — Architecture & Implementation Plan

## 1. Overview & Goals

**Purpose.** An enterprise multi-agent platform that analyzes call audio and its transcript to detect verbal abuse, threats, fraud/social engineering, and compliance violations; correlates audio- and transcript-level findings into a single risk assessment; raises supervisor alerts; and exposes results via a reporting layer, REST API, and dashboard.

**Goals**
- Detect threats using both **audio** signals (prosody, emotion, background context) and **transcript** signals (language-based detection across four categories), then correlate them rather than trusting either channel alone.
- Produce a single, explainable **risk score** and structured findings per call, traceable back to the specific agent(s) and evidence (transcript spans / audio segments) that drove it.
- Be measurable: the platform must be evaluable end-to-end and per-agent against Project 1's (`synthetic_threat_data_generator`) labeled benchmark releases.
- Start with batch (post-call) processing; be architected so streaming/live-call analysis is a later extension, not a rewrite.

**Non-goals (Phase 1)**
- Live/streaming call analysis is out of scope for the initial build (see §13 Phasing) — Phase 1 processes completed call recordings.
- Dashboard frontend implementation detail is out of scope for this document; the platform commits only to the REST API contract the dashboard will consume.
- No commitment to a specific cloud provider or specific LLM/STT/TTS vendor — see §10.

## 2. Architecture Diagram

```
                    Dataset Generator (Project 1, for eval)
                     or Real Call Ingestion (production)
                                  │
                                  ▼
                         Call Audio + (optional) Reference Transcript
                                  │
                 ┌────────────────┴────────────────┐
                 ▼                                  ▼
        Audio Intelligence Domain          Transcript Intelligence Domain
     (STT, Diarization, Prosody,           (Verbal Abuse, Threat, Fraud &
      Emotion, Background Audio)            Social Eng., Compliance)
                 │                                  │
                 ▼                                  ▼
        Audio Correlation Agent           Transcript Correlation Agent
                 └────────────────┬────────────────┘
                                  ▼
                  Threat Correlation & Decision Agent
                                  ▼
                  Risk Score + Findings + Alert Decision
                                  │
              ┌───────────────────┼───────────────────┐
              ▼                   ▼                   ▼
        Supervisor Alerts    Reporting            REST API → Dashboard
```

## 3. Agent Catalog (12 agents, 3 domains)

### 3.1 Audio Intelligence Domain

| Agent | Input | Output |
|---|---|---|
| **Speech-to-Text Agent** | raw call audio | transcript with word-level timestamps, via pluggable `STTClient` |
| **Speaker Diarization Agent** | raw call audio | speaker-labeled time segments ("who spoke when") |
| **Prosody Analysis Agent** | raw call audio (+ diarization) | per-segment pitch, pace, volume, pause patterns; stress/aggression indicators |
| **Emotion Detection Agent** | raw call audio (+ diarization) | per-segment emotion classification (anger, fear, distress, calm, ...) |
| **Background Audio Detection Agent** | raw call audio | detected background events (crowd noise, other voices, environment cues) with timestamps |
| **Audio Correlation Agent** | outputs of the 5 agents above | unified audio intelligence summary + audio domain score, with supporting evidence per finding |

### 3.2 Transcript Intelligence Domain

| Agent | Input | Output |
|---|---|---|
| **Verbal Abuse Detection Agent** | transcript (+ diarization) | flagged spans of abusive/harassing language, severity |
| **Threat Detection Agent** | transcript | flagged spans indicating explicit/implicit threats of harm, severity |
| **Fraud & Social Engineering Agent** | transcript | flagged spans indicating scam/manipulation patterns, severity |
| **Compliance Detection Agent** | transcript | flagged spans/gaps indicating policy or regulatory violations (missing disclosures, PII/PCI exposure), severity |
| **Transcript Correlation Agent** | outputs of the 4 agents above | unified transcript intelligence summary + transcript domain score, with supporting evidence per finding |

### 3.3 Enterprise Decision Domain

| Agent | Input | Output |
|---|---|---|
| **Threat Correlation & Decision Agent** | Audio Correlation output + Transcript Correlation output | overall risk score, final category/severity determination, alert decision, structured findings package for reporting/API/dashboard |

This is deliberately the exact 12-agent, 3-domain breakdown from the SRS — no agents added or removed.

## 4. LangGraph Orchestration Design

### 4.1 `CallState`

The shared state object threaded through the graph:

```
CallState
  call_id: str
  audio_ref: str                      # object-store pointer to source audio
  transcript: Transcript | None        # populated by STT Agent
  diarization: DiarizationResult | None
  audio_findings: dict[str, AgentFinding]      # keyed by agent name
  transcript_findings: dict[str, AgentFinding]
  audio_domain_score: DomainScore | None
  transcript_domain_score: DomainScore | None
  final_assessment: ThreatAssessment | None
```

### 4.2 Graph structure

- **Audio subgraph:** Speech-to-Text and Speaker Diarization run in parallel off the raw audio, then merge into a diarized transcript. Prosody, Emotion, and Background Audio Detection run in parallel directly off the raw audio (diarization-aware where useful, not blocking on STT). All five join at **Audio Correlation**.
- **Transcript subgraph:** once a transcript is available, the four transcript detection agents run in parallel, joining at **Transcript Correlation**.
- **Top-level graph:** the audio subgraph and transcript subgraph both execute (transcript subgraph depends on STT output from the audio subgraph, so it is not fully independent — it starts as soon as the transcript is ready rather than waiting for the rest of the audio subgraph). Both correlation outputs join at the **Threat Correlation & Decision Agent**, which produces the `final_assessment`.
- Each agent node is a plain function/tool call against `CallState`, making unit-testing an agent (given a fixture `CallState`) straightforward without running the full graph.

## 5. Risk Scoring Model

- Each detection agent emits findings with a **severity** and **confidence**, not just a binary flag.
- Each Correlation Agent (Audio, Transcript) reduces its domain's findings into a single **domain score** (0–100) via a configurable weighted function of severity × confidence across its agents.
- The Decision Agent combines `audio_domain_score` and `transcript_domain_score` into an overall risk score using configurable **per-category weights** (e.g., a `threat_of_violence` transcript finding corroborated by high-arousal `Emotion`/`Prosody` audio findings should score higher than the same transcript finding with a calm-audio profile — corroboration across domains is the point of correlation, not just averaging).
- **Calibration:** weights and thresholds are tuned by running the Phase 1 pipeline against Project 1 benchmark releases and comparing predicted risk scores/categories to ground truth (precision/recall per category, score distribution vs. severity). This is the evaluation loop described in §12 — calibration is data-driven, not hand-picked.

## 6. Supervisor Alerting

- The Decision Agent's output includes an alert decision (`alert` / `no_alert`) based on configurable risk-score and category thresholds.
- Alert payload: call ID, risk score, category/severity, top contributing findings with evidence (transcript spans / audio timestamps), timestamp.
- Delivery is behind a `NotificationChannel` interface (webhook-based); concrete channels (Slack, email, PagerDuty, etc.) are adapters implementing that interface — the Decision Agent never depends on a specific channel.

## 7. Reporting & Dashboard

- A **Reporting** service aggregates assessments over time: trend reports, per-category volumes, compliance summaries — reads from the same data store the API uses, no separate pipeline.
- The **Dashboard** is treated as a thin consumer of the REST API (§8); its frontend stack is an explicit open decision deferred to a later planning pass, not detailed here.

## 8. REST API

FastAPI service exposing:
- `POST /calls` — submit a call recording for analysis (enqueues through the graph).
- `GET /calls/{call_id}` — fetch assessment/status for a call.
- `GET /alerts` — list/query raised alerts.
- `GET /calls` — list/query calls with filters (category, severity, date range).
- `POST /evaluation/run` — run the platform against a specified Project 1 benchmark release and return per-agent + end-to-end metrics (the evaluation harness, exposed as an API for CI/regression use as well as ad hoc use).

## 9. Data & Storage

- **Postgres** for call records, findings, assessments, alerts — relational, cloud-agnostic, easy to containerize locally and run managed anywhere in production.
- **S3-compatible object storage** (MinIO locally; S3/Azure Blob/GCS in production via the same interface) for audio files and any large artifacts.
- Both accessed through a `storage/` abstraction layer so no agent or API code talks to a specific backend directly.

## 10. Provider Abstraction

Same design principle as Project 1:

```
LLMClient (interface)     # used by any agent doing LLM-based classification/reasoning
  generate(prompt, **kwargs) -> str

STTClient (interface)     # used by Speech-to-Text Agent
  transcribe(audio) -> Transcript   # with word-level timestamps
```

Concrete backends selected via config, under `src/providers/`. `LLMClient` here is the same interface shape as Project 1's — once both projects are stable, extracting it into a shared internal package is worth revisiting (not done now, to avoid coupling two independently-evolving projects prematurely).

## 11. Module / Directory Structure

```
threat_detection_system/
  src/
    agents/
      audio/
        stt_agent.py
        diarization_agent.py
        prosody_agent.py
        emotion_agent.py
        background_audio_agent.py
        audio_correlation_agent.py
      transcript/
        verbal_abuse_agent.py
        threat_agent.py
        fraud_social_engineering_agent.py
        compliance_agent.py
        transcript_correlation_agent.py
      decision/
        threat_correlation_decision_agent.py
    orchestration/
      graph.py             # top-level LangGraph graph
      audio_subgraph.py
      transcript_subgraph.py
      state.py              # CallState definition
    providers/
      llm_client.py          # interface
      llm_<vendor>.py
      stt_client.py           # interface
      stt_<vendor>.py
    risk_scoring/
      scoring_engine.py
      weights.py
    alerting/
      notification_channel.py   # interface
      channels/                  # webhook/email/slack adapters
    reporting/
      report_generator.py
    api/
      main.py                    # FastAPI app
      routers/
        calls.py
        alerts.py
        evaluation.py
    storage/
      db.py                       # Postgres access
      object_store_client.py      # S3-compatible
    evaluation/
      harness.py                  # consumes Project 1 manifest.jsonl, scores agents/end-to-end
      metrics.py
  configs/
    risk_scoring.yaml
    alert_thresholds.yaml
    providers.yaml
  tests/
  docs/
    architecture-plan.md   # this file
```

## 12. Evaluation Loop (integration with Project 1)

This is the explicit contract between the two projects:

1. The evaluation harness (`src/evaluation/harness.py`) reads a Project 1 `manifest.jsonl` (see `synthetic_threat_data_generator/docs/architecture-plan.md` §7).
2. For each sample, it runs the full graph against `audio.wav`, producing a `final_assessment`.
3. It compares agent-level findings and the final assessment against the sample's `ground_truth.json` (§4 of Project 1's doc — the `GroundTruthLabel` schema), computing:
   - Per-transcript-agent precision/recall/F1 against `category`/`severity`/span overlap.
   - Per-audio-agent accuracy against `expected_emotion` / `expected_background_event` / `expected_prosody_notes`.
   - End-to-end risk-score calibration against severity, and alert-decision precision/recall against whether a sample should have alerted.
4. Emits a metrics report (also servable via `POST /evaluation/run`), which is the basis for risk-scoring calibration (§5) and regression testing across code/prompt changes.

**Consistency requirement:** any change to the `GroundTruthLabel` schema in Project 1 must be reflected in this harness's parsing logic, and vice versa — these two are versioned together even though they live in separate project directories.

## 13. Phasing

- **Phase 1 — Batch:** full graph running post-call on a recorded audio file; REST API for submit/fetch; evaluation harness wired against Project 1 releases; alerting and reporting functional against completed-call output.
- **Phase 2 — Streaming extension (future work, not built in Phase 1):**
  - Replace batch STT with a streaming `STTClient` variant; diarization becomes incremental.
  - Detection agents move from "run once on final transcript" to sliding-window re-evaluation on growing transcript/audio.
  - Decision Agent needs alert de-duplication/debouncing (don't re-alert on the same escalating finding every window).
  - Supervisor alert delivery becomes live/push rather than post-hoc, with latency budgets per agent.
  - This phase is explicitly deferred so Phase 1 isn't over-built for a requirement that isn't being implemented yet.

## 14. Non-Functional Considerations

- **Observability:** per-agent tracing/logging of inputs, outputs, latency, and provider calls within each graph run, keyed by `call_id`, so any assessment is explainable after the fact.
- **Testing strategy:** unit tests per agent against fixture `CallState`s; an integration test running the full graph against a small fixture set (a handful of Project 1 samples checked into `tests/fixtures/`); the evaluation harness itself doubles as a regression suite against a full benchmark release.
- **Scalability:** agents are stateless workers reading/writing `CallState` — horizontally scalable via containers; Postgres and object storage are the only shared state, both externalizable to managed services in production without code changes.
