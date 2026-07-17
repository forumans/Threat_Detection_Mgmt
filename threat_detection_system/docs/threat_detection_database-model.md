# Threat Detection System — Database Model

Companion to `threat_detection_architecture-plan.md`. Defines the Postgres schema and answers whether a vector database is needed.

## 1. Design Principles

- **Runs, not just calls.** A `call` is ingested once (audio + static metadata), but the agent graph may execute against it multiple times — the initial production run, a re-processing run after a model/prompt update, or an evaluation run against a benchmark sample. All agent-produced data (transcript, findings, scores, assessment) is keyed off `pipeline_run_id`, not directly off `call_id`. This preserves history across reprocessing and is what the evaluation harness (architecture doc §12) needs to compare multiple runs against the same ground truth.
- **One findings table, not nine.** All non-STT audio and transcript agents write into a single `agent_findings` table, distinguished by `agent_name`/`domain`. The Correlation agents already consume findings generically (architecture doc §3.1–3.2); a single table avoids a bespoke schema per agent for what is structurally the same record (severity, confidence, evidence, summary).
- **Multi-tenant via `org_id` + Row-Level Security**, on every production-data table. Benchmark/evaluation tables are deliberately **not** tenant-scoped — synthetic benchmark data and evaluation metrics are an internal engineering concern shared across the platform, not customer data.
- **The vector DB is referenced, not embedded.** Postgres stores the source-of-truth text/metadata and a pointer (`vector_ref`) into the external vector database; the vector DB stores only the embedding plus the minimal payload needed for ANN filtering. Postgres remains the system of record.

## 2. Core Tables

```sql
organizations
  org_id UUID PK
  name TEXT
  status TEXT               -- active | suspended
  settings JSONB
  created_at TIMESTAMPTZ

calls
  call_id UUID PK
  org_id UUID FK -> organizations
  source TEXT                -- production_ingestion | benchmark_evaluation
  benchmark_sample_ref TEXT NULL   -- Project 1 sample_id, when source = benchmark_evaluation
  audio_object_key TEXT       -- pointer into S3-compatible object store
  duration_ms INT
  channel_info JSONB
  locale TEXT
  ingested_at TIMESTAMPTZ
  created_at TIMESTAMPTZ
  INDEX (org_id, ingested_at)

pipeline_runs
  run_id UUID PK
  call_id UUID FK -> calls
  org_id UUID FK -> organizations          -- denormalized so RLS doesn't require a join
  trigger_type TEXT           -- production | reprocessing | evaluation
  benchmark_release_id UUID NULL FK -> benchmark_releases
  graph_version TEXT           -- git sha / semver of the LangGraph definition used
  status TEXT                  -- pending | running | completed | failed
  started_at TIMESTAMPTZ
  completed_at TIMESTAMPTZ NULL
  error_detail JSONB NULL
  INDEX (call_id, started_at)
  INDEX (org_id, status)
```

## 3. Transcript & Audio-Analysis Tables (per `run_id`)

```sql
stt_words
  word_id UUID PK
  run_id UUID FK -> pipeline_runs
  start_ms INT
  end_ms INT
  text TEXT
  confidence NUMERIC(4,3)

diarization_segments
  segment_id UUID PK
  run_id UUID FK -> pipeline_runs
  speaker_label TEXT            -- e.g. "speaker_1"
  start_ms INT
  end_ms INT
  confidence NUMERIC(4,3)

transcript_turns                 -- merged/diarized transcript consumed by transcript agents
  turn_id UUID PK
  run_id UUID FK -> pipeline_runs
  turn_index INT
  speaker_label TEXT
  start_ms INT
  end_ms INT
  text TEXT
  UNIQUE (run_id, turn_index)
```

## 4. Findings, Scores, and Final Assessment

```sql
agent_findings
  finding_id UUID PK
  run_id UUID FK -> pipeline_runs
  org_id UUID FK -> organizations                -- denormalized for RLS
  domain TEXT               -- audio | transcript
  agent_name TEXT            -- prosody | emotion | background_audio | verbal_abuse |
                              -- threat | fraud_social_engineering | compliance
  category TEXT NULL         -- verbal_abuse | threat_of_violence | fraud_social_engineering |
                              -- compliance_violation  (NULL for pure audio-signal agents like prosody)
  severity TEXT NULL         -- low | medium | high | critical
  confidence NUMERIC(4,3)
  evidence JSONB             -- e.g. {"turn_id":..,"char_start":..,"char_end":..} or {"start_ms":..,"end_ms":..}
  summary TEXT
  created_at TIMESTAMPTZ
  INDEX (run_id, domain)
  INDEX (org_id, category, severity)

domain_scores
  domain_score_id UUID PK
  run_id UUID FK -> pipeline_runs
  domain TEXT                -- audio | transcript
  score NUMERIC(5,2)
  summary TEXT
  created_at TIMESTAMPTZ
  UNIQUE (run_id, domain)

domain_score_findings           -- join: which findings fed a domain score
  domain_score_id UUID FK -> domain_scores
  finding_id UUID FK -> agent_findings
  PRIMARY KEY (domain_score_id, finding_id)

threat_assessments
  assessment_id UUID PK
  run_id UUID FK -> pipeline_runs UNIQUE      -- one assessment per run
  org_id UUID FK -> organizations
  audio_domain_score_id UUID FK -> domain_scores
  transcript_domain_score_id UUID FK -> domain_scores
  risk_score NUMERIC(5,2)
  final_category TEXT NULL
  final_severity TEXT NULL
  alert_decision BOOLEAN
  decision_summary TEXT
  created_at TIMESTAMPTZ
  INDEX (org_id, risk_score)
```

## 5. Alerts & Notifications

```sql
alerts
  alert_id UUID PK
  org_id UUID FK -> organizations
  call_id UUID FK -> calls
  run_id UUID FK -> pipeline_runs
  assessment_id UUID FK -> threat_assessments
  risk_score NUMERIC(5,2)
  category TEXT
  severity TEXT
  status TEXT                 -- open | acknowledged | resolved | dismissed
  assigned_user_id TEXT NULL    -- opaque external identity; no local users table
  created_at TIMESTAMPTZ
  acknowledged_at TIMESTAMPTZ NULL
  resolved_at TIMESTAMPTZ NULL
  INDEX (org_id, status, created_at)

alert_findings                   -- join: top contributing findings shown in the alert payload
  alert_id UUID FK -> alerts
  finding_id UUID FK -> agent_findings
  PRIMARY KEY (alert_id, finding_id)

notification_deliveries
  delivery_id UUID PK
  alert_id UUID FK -> alerts
  channel_type TEXT            -- webhook | email | slack | pagerduty
  destination TEXT
  status TEXT                   -- pending | sent | failed
  attempted_at TIMESTAMPTZ
  response_meta JSONB NULL
```

## 6. Benchmark & Evaluation Tables (global — not org-scoped)

```sql
benchmark_releases
  release_id UUID PK
  dataset_version TEXT
  manifest_uri TEXT             -- pointer to Project 1's manifest.jsonl
  coverage_report JSONB
  imported_at TIMESTAMPTZ

evaluation_runs
  eval_run_id UUID PK
  release_id UUID FK -> benchmark_releases
  triggered_by TEXT              -- ci | manual
  status TEXT
  started_at TIMESTAMPTZ
  completed_at TIMESTAMPTZ NULL
  metrics_summary JSONB           -- per-agent + end-to-end precision/recall/F1, score calibration

evaluation_sample_results
  eval_sample_result_id UUID PK
  eval_run_id UUID FK -> evaluation_runs
  benchmark_sample_id TEXT         -- Project 1 sample_id
  call_id UUID FK -> calls
  run_id UUID FK -> pipeline_runs
  result_detail JSONB               -- per-agent predicted vs. ground_truth comparison
```

These tables intentionally have no `org_id` — evaluation is an internal engineering process against synthetic data, not customer data, so it isn't subject to tenant isolation.

## 7. Vector-DB-Linked Metadata Tables

```sql
policy_documents
  policy_doc_id UUID PK
  org_id UUID FK -> organizations       -- compliance rules are org-specific
  title TEXT
  source_uri TEXT
  version TEXT
  content_hash TEXT
  ingested_at TIMESTAMPTZ

policy_document_chunks
  chunk_id UUID PK
  policy_doc_id UUID FK -> policy_documents
  chunk_index INT
  text TEXT
  vector_ref TEXT              -- point ID in the external vector DB
  UNIQUE (policy_doc_id, chunk_index)

call_embeddings
  call_embedding_id UUID PK
  org_id UUID FK -> organizations
  call_id UUID FK -> calls
  run_id UUID FK -> pipeline_runs
  embedding_type TEXT           -- call_summary | alert_summary
  vector_ref TEXT                -- point ID in the external vector DB
  created_at TIMESTAMPTZ
```

## 8. Multi-Tenancy Enforcement

- Every production-data table carries `org_id`, denormalized onto child tables (e.g. `agent_findings`, `pipeline_runs`) even though it's reachable via `call_id`, specifically so RLS policies don't require a join.
- Postgres Row-Level Security policies of the form `USING (org_id = current_setting('app.current_org_id')::uuid)` on each tenant-scoped table; the API layer sets `app.current_org_id` per request/session (from the external IdP's token claims).
- `benchmark_releases`, `evaluation_runs`, `evaluation_sample_results` are exempt from RLS — internal/global, not customer data.

## 9. ER Overview

```
organizations 1───* calls 1───* pipeline_runs 1───* agent_findings
                                    │                      │
                                    │                      └──* domain_score_findings ──* domain_scores
                                    ├───1 threat_assessments ───* alerts ───* notification_deliveries
                                    ├───* transcript_turns
                                    ├───* stt_words
                                    └───* diarization_segments

benchmark_releases 1───* evaluation_runs 1───* evaluation_sample_results ──> calls / pipeline_runs

organizations 1───* policy_documents 1───* policy_document_chunks (→ vector DB)
organizations 1───* calls ───* call_embeddings (→ vector DB)
```

## 10. Do You Need a Vector DB?

**Yes** — the two confirmed use cases are real requirements, not speculative:
1. **Compliance RAG** — the Compliance Detection Agent needs to retrieve relevant clauses from an organization's policy/regulation corpus to check a transcript against, rather than relying on an LLM's unaided knowledge of that org's specific policies.
2. **Similarity search** — investigators/supervisors need to find calls or alerts similar to a given one from the dashboard (e.g., "show me past alerts like this").

**Recommendation: a dedicated, self-hostable vector database — Qdrant as the concrete default** — rather than pgvector-in-Postgres or a managed SaaS vector DB:
- Stays consistent with the platform's existing cloud-agnostic/containerized deployment decision (architecture doc, Deployment) — Qdrant runs as a container alongside Postgres/MinIO.
- Native payload filtering lets vector search enforce `org_id` scoping the same way RLS does for Postgres, which a pgvector approach would need to hand-roll via `WHERE` clauses anyway (so pgvector's "one less moving part" advantage is smaller here than usual).
- Accessed through a `VectorStoreClient` interface — the same abstraction pattern already used for `LLMClient`/`STTClient` (architecture doc §10) — so Qdrant is a swappable default, not a hard dependency; Milvus/Weaviate remain viable alternatives behind the same interface.

Two collections:
- `policy_chunks` — embeddings of `policy_document_chunks.text`; payload carries `org_id`, `policy_doc_id`, `chunk_id`.
- `call_summaries` — embeddings of call/alert summaries; payload carries `org_id`, `call_id`, `embedding_type`.

**Sequencing note:** the core detection pipeline (Phase 1, batch — architecture doc §13) does not depend on the vector DB at all. It only becomes load-bearing once the Compliance Agent's RAG behavior and dashboard similarity-search features are actually built. Stand up Qdrant when those features start, not as day-one Phase 1 infrastructure.
