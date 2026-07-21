-- ============================================================================
-- Threat Detection System — Database DDL
--
-- Generated from: threat_detection_system/docs/database/threat_detection_database-model.md
-- Target: PostgreSQL 13+
--
-- To provision from scratch (run as a superuser / role with CREATEDB):
--
--   psql -h <host> -U <superuser> -d postgres -f 00_create_database.sql
--   psql -h <host> -U <superuser> -d threat_detection -f schema.sql
--
-- This script expects to be run while already connected to the
-- "threat_detection" database created by 00_create_database.sql.
-- It is idempotent (safe to re-run) via IF NOT EXISTS guards.
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 0. Extensions
-- ----------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS pgcrypto;   -- gen_random_uuid()


-- ============================================================================
-- 1. Core tables (organizations, calls, pipeline_runs)
-- ============================================================================

CREATE TABLE IF NOT EXISTS organizations (
    org_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name       TEXT NOT NULL,
    status     TEXT NOT NULL CHECK (status IN ('active', 'suspended')),
    settings   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE organizations IS 'Tenant root. Every production-data table is scoped to an org_id via RLS.';

CREATE TABLE IF NOT EXISTS benchmark_releases (
    release_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_version TEXT NOT NULL,
    manifest_uri    TEXT NOT NULL,   -- pointer to Project 1's manifest.jsonl
    coverage_report JSONB,
    imported_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON TABLE benchmark_releases IS 'Global (not org-scoped) — synthetic benchmark data is not customer data.';

CREATE TABLE IF NOT EXISTS calls (
    call_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id               UUID NOT NULL REFERENCES organizations(org_id),
    source               TEXT NOT NULL CHECK (source IN ('production_ingestion', 'benchmark_evaluation')),
    benchmark_sample_ref TEXT,             -- Project 1 sample_id, when source = benchmark_evaluation
    audio_object_key     TEXT NOT NULL,    -- pointer into S3-compatible object store
    duration_ms          INT,
    channel_info         JSONB,
    locale               TEXT,
    ingested_at          TIMESTAMPTZ NOT NULL,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_calls_org_ingested ON calls (org_id, ingested_at);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    call_id               UUID NOT NULL REFERENCES calls(call_id),
    org_id                UUID NOT NULL REFERENCES organizations(org_id),   -- denormalized: avoids a join for RLS
    trigger_type          TEXT NOT NULL CHECK (trigger_type IN ('production', 'reprocessing', 'evaluation')),
    benchmark_release_id  UUID REFERENCES benchmark_releases(release_id),
    graph_version         TEXT NOT NULL,    -- git sha / semver of the LangGraph definition used
    status                TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at            TIMESTAMPTZ NOT NULL,
    completed_at          TIMESTAMPTZ,
    error_detail          JSONB
);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_call_started ON pipeline_runs (call_id, started_at);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_org_status ON pipeline_runs (org_id, status);
CREATE INDEX IF NOT EXISTS ix_pipeline_runs_benchmark_release ON pipeline_runs (benchmark_release_id);


-- ============================================================================
-- 2. Transcript & audio-analysis tables (per run_id)
-- ============================================================================

CREATE TABLE IF NOT EXISTS stt_words (
    word_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id     UUID NOT NULL REFERENCES pipeline_runs(run_id),
    start_ms   INT NOT NULL,
    end_ms     INT NOT NULL,
    text       TEXT NOT NULL,
    confidence NUMERIC(4,3)
);
CREATE INDEX IF NOT EXISTS ix_stt_words_run ON stt_words (run_id);

CREATE TABLE IF NOT EXISTS diarization_segments (
    segment_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES pipeline_runs(run_id),
    speaker_label TEXT NOT NULL,   -- e.g. "speaker_1"
    start_ms      INT NOT NULL,
    end_ms        INT NOT NULL,
    confidence    NUMERIC(4,3)
);
CREATE INDEX IF NOT EXISTS ix_diarization_segments_run ON diarization_segments (run_id);

CREATE TABLE IF NOT EXISTS transcript_turns (
    turn_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id        UUID NOT NULL REFERENCES pipeline_runs(run_id),
    turn_index    INT NOT NULL,
    speaker_label TEXT NOT NULL,
    start_ms      INT NOT NULL,
    end_ms        INT NOT NULL,
    text          TEXT NOT NULL,
    UNIQUE (run_id, turn_index)
);
COMMENT ON TABLE transcript_turns IS 'Merged/diarized transcript consumed by the transcript-intelligence agents.';
CREATE INDEX IF NOT EXISTS ix_transcript_turns_run ON transcript_turns (run_id);


-- ============================================================================
-- 3. Findings, scores, and final assessment
-- ============================================================================

CREATE TABLE IF NOT EXISTS agent_findings (
    finding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id     UUID NOT NULL REFERENCES pipeline_runs(run_id),
    org_id     UUID NOT NULL REFERENCES organizations(org_id),   -- denormalized for RLS
    domain     TEXT NOT NULL CHECK (domain IN ('audio', 'transcript')),
    agent_name TEXT NOT NULL CHECK (agent_name IN (
                   'prosody', 'emotion', 'background_audio',
                   'verbal_abuse', 'threat', 'fraud_social_engineering', 'compliance'
               )),
    category   TEXT CHECK (category IS NULL OR category IN (
                   'verbal_abuse', 'threat_of_violence', 'fraud_social_engineering', 'compliance_violation'
               )),
    severity   TEXT CHECK (severity IS NULL OR severity IN ('low', 'medium', 'high', 'critical')),
    confidence NUMERIC(4,3) NOT NULL,
    evidence   JSONB NOT NULL,   -- {"turn_id":..,"char_start":..,"char_end":..} or {"start_ms":..,"end_ms":..}
    summary    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
COMMENT ON COLUMN agent_findings.category IS 'NULL for pure audio-signal agents that do not classify into the threat taxonomy (e.g. prosody).';
CREATE INDEX IF NOT EXISTS ix_agent_findings_run_domain ON agent_findings (run_id, domain);
CREATE INDEX IF NOT EXISTS ix_agent_findings_org_category_severity ON agent_findings (org_id, category, severity);

CREATE TABLE IF NOT EXISTS domain_scores (
    domain_score_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id          UUID NOT NULL REFERENCES pipeline_runs(run_id),
    domain          TEXT NOT NULL CHECK (domain IN ('audio', 'transcript')),
    score           NUMERIC(5,2) NOT NULL,
    summary         TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, domain)
);

CREATE TABLE IF NOT EXISTS domain_score_findings (
    domain_score_id UUID NOT NULL REFERENCES domain_scores(domain_score_id),
    finding_id      UUID NOT NULL REFERENCES agent_findings(finding_id),
    PRIMARY KEY (domain_score_id, finding_id)
);
CREATE INDEX IF NOT EXISTS ix_domain_score_findings_finding ON domain_score_findings (finding_id);

CREATE TABLE IF NOT EXISTS threat_assessments (
    assessment_id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id                     UUID NOT NULL UNIQUE REFERENCES pipeline_runs(run_id),   -- one assessment per run
    org_id                     UUID NOT NULL REFERENCES organizations(org_id),
    audio_domain_score_id      UUID NOT NULL REFERENCES domain_scores(domain_score_id),
    transcript_domain_score_id UUID NOT NULL REFERENCES domain_scores(domain_score_id),
    risk_score                 NUMERIC(5,2) NOT NULL,
    final_category              TEXT CHECK (final_category IS NULL OR final_category IN (
                                    'verbal_abuse', 'threat_of_violence', 'fraud_social_engineering', 'compliance_violation'
                                )),
    final_severity              TEXT CHECK (final_severity IS NULL OR final_severity IN ('low', 'medium', 'high', 'critical')),
    alert_decision               BOOLEAN NOT NULL,
    decision_summary             TEXT NOT NULL,
    created_at                   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_threat_assessments_org_risk ON threat_assessments (org_id, risk_score);


-- ============================================================================
-- 4. Alerts & notifications
-- ============================================================================

CREATE TABLE IF NOT EXISTS alerts (
    alert_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            UUID NOT NULL REFERENCES organizations(org_id),
    call_id           UUID NOT NULL REFERENCES calls(call_id),
    run_id            UUID NOT NULL REFERENCES pipeline_runs(run_id),
    assessment_id     UUID NOT NULL REFERENCES threat_assessments(assessment_id),
    risk_score        NUMERIC(5,2) NOT NULL,
    category          TEXT NOT NULL CHECK (category IN (
                          'verbal_abuse', 'threat_of_violence', 'fraud_social_engineering', 'compliance_violation'
                      )),
    severity          TEXT NOT NULL CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    status            TEXT NOT NULL CHECK (status IN ('open', 'acknowledged', 'resolved', 'dismissed')),
    assigned_user_id  TEXT,   -- opaque external identity; no local users table
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    acknowledged_at   TIMESTAMPTZ,
    resolved_at       TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS ix_alerts_org_status_created ON alerts (org_id, status, created_at);
CREATE INDEX IF NOT EXISTS ix_alerts_call ON alerts (call_id);
CREATE INDEX IF NOT EXISTS ix_alerts_run ON alerts (run_id);

CREATE TABLE IF NOT EXISTS alert_findings (
    alert_id   UUID NOT NULL REFERENCES alerts(alert_id),
    finding_id UUID NOT NULL REFERENCES agent_findings(finding_id),
    PRIMARY KEY (alert_id, finding_id)
);
CREATE INDEX IF NOT EXISTS ix_alert_findings_finding ON alert_findings (finding_id);

CREATE TABLE IF NOT EXISTS notification_deliveries (
    delivery_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    alert_id      UUID NOT NULL REFERENCES alerts(alert_id),
    channel_type  TEXT NOT NULL CHECK (channel_type IN ('webhook', 'email', 'slack', 'pagerduty')),
    destination   TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('pending', 'sent', 'failed')),
    attempted_at  TIMESTAMPTZ NOT NULL,
    response_meta JSONB
);
CREATE INDEX IF NOT EXISTS ix_notification_deliveries_alert ON notification_deliveries (alert_id);


-- ============================================================================
-- 5. Benchmark & evaluation tables (global — not org-scoped)
-- ============================================================================

CREATE TABLE IF NOT EXISTS evaluation_runs (
    eval_run_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    release_id       UUID NOT NULL REFERENCES benchmark_releases(release_id),
    triggered_by     TEXT NOT NULL CHECK (triggered_by IN ('ci', 'manual')),
    status           TEXT NOT NULL CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    started_at       TIMESTAMPTZ NOT NULL,
    completed_at     TIMESTAMPTZ,
    metrics_summary  JSONB   -- per-agent + end-to-end precision/recall/F1, score calibration
);
CREATE INDEX IF NOT EXISTS ix_evaluation_runs_release ON evaluation_runs (release_id);

CREATE TABLE IF NOT EXISTS evaluation_sample_results (
    eval_sample_result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id           UUID NOT NULL REFERENCES evaluation_runs(eval_run_id),
    benchmark_sample_id   TEXT NOT NULL,   -- Project 1 sample_id
    call_id               UUID NOT NULL REFERENCES calls(call_id),
    run_id                UUID NOT NULL REFERENCES pipeline_runs(run_id),
    result_detail         JSONB NOT NULL   -- per-agent predicted vs. ground_truth comparison
);
CREATE INDEX IF NOT EXISTS ix_eval_sample_results_eval_run ON evaluation_sample_results (eval_run_id);
CREATE INDEX IF NOT EXISTS ix_eval_sample_results_call ON evaluation_sample_results (call_id);
CREATE INDEX IF NOT EXISTS ix_eval_sample_results_run ON evaluation_sample_results (run_id);

COMMENT ON TABLE evaluation_runs IS 'Global (not org-scoped) — internal engineering process against synthetic data.';
COMMENT ON TABLE evaluation_sample_results IS 'Global (not org-scoped) — internal engineering process against synthetic data.';


-- ============================================================================
-- 6. Vector-DB-linked metadata tables (compliance RAG + similarity search)
-- ============================================================================

CREATE TABLE IF NOT EXISTS policy_documents (
    policy_doc_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id        UUID NOT NULL REFERENCES organizations(org_id),   -- compliance rules are org-specific
    title         TEXT NOT NULL,
    source_uri    TEXT NOT NULL,
    version       TEXT NOT NULL,
    content_hash  TEXT NOT NULL,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_policy_documents_org ON policy_documents (org_id);

CREATE TABLE IF NOT EXISTS policy_document_chunks (
    chunk_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    policy_doc_id UUID NOT NULL REFERENCES policy_documents(policy_doc_id),
    chunk_index   INT NOT NULL,
    text          TEXT NOT NULL,
    vector_ref    TEXT NOT NULL,   -- point ID in the external vector DB (e.g. Qdrant)
    UNIQUE (policy_doc_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS ix_policy_document_chunks_doc ON policy_document_chunks (policy_doc_id);

CREATE TABLE IF NOT EXISTS call_embeddings (
    call_embedding_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id            UUID NOT NULL REFERENCES organizations(org_id),
    call_id           UUID NOT NULL REFERENCES calls(call_id),
    run_id            UUID NOT NULL REFERENCES pipeline_runs(run_id),
    embedding_type    TEXT NOT NULL CHECK (embedding_type IN ('call_summary', 'alert_summary')),
    vector_ref        TEXT NOT NULL,   -- point ID in the external vector DB (e.g. Qdrant)
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_call_embeddings_org ON call_embeddings (org_id);
CREATE INDEX IF NOT EXISTS ix_call_embeddings_call ON call_embeddings (call_id);
CREATE INDEX IF NOT EXISTS ix_call_embeddings_run ON call_embeddings (run_id);


-- ============================================================================
-- 7. Row-Level Security (multi-tenant enforcement)
--
-- Applied to every table that carries org_id directly. Tables without org_id
-- (e.g. transcript_turns, stt_words, domain_scores, alert_findings) are
-- reached only through an already-scoped parent (run_id/call_id/alert_id)
-- and are not directly RLS-protected, per the design doc.
--
-- The application must SET app.current_org_id per request/session, e.g.:
--   SET LOCAL app.current_org_id = '<org-uuid-from-auth-token>';
-- ============================================================================

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'calls', 'pipeline_runs', 'agent_findings', 'threat_assessments',
        'alerts', 'policy_documents', 'call_embeddings'
    ]
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
        EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
        EXECUTE format('DROP POLICY IF EXISTS %I ON %I', t || '_tenant_isolation', t);
        EXECUTE format(
            'CREATE POLICY %I ON %I USING (org_id = current_setting(''app.current_org_id'', true)::uuid)',
            t || '_tenant_isolation', t
        );
    END LOOP;
END $$;
