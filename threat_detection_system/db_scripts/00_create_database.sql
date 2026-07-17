-- ============================================================================
-- Threat Detection System — Database Creation
--
-- Run this FIRST, connected to the default "postgres" maintenance database,
-- as a role with CREATEDB privilege. CREATE DATABASE cannot run inside a
-- transaction block, so it must live in its own script/connection — separate
-- from schema.sql, which creates tables inside the new database and expects
-- to be run against it directly.
--
-- Usage:
--   psql -h <host> -U <superuser> -d postgres -f 00_create_database.sql
--   psql -h <host> -U <superuser> -d threat_detection -f schema.sql
--
-- Idempotent: uses \gexec to only issue CREATE DATABASE if it doesn't
-- already exist (CREATE DATABASE has no IF NOT EXISTS clause in Postgres).
-- ============================================================================

SELECT 'CREATE DATABASE threat_detection ENCODING ''UTF8'' TEMPLATE template0'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'threat_detection')
\gexec

COMMENT ON DATABASE threat_detection IS 'Threat Detection System — Project 2 (see threat_detection_system/docs).';

-- ----------------------------------------------------------------------------
-- Optional: dedicated application role for the platform to connect as.
-- Uncomment and set a real password (or manage this role via your secrets
-- tooling / IaC instead) before running in a real environment. A non-
-- superuser role is required for the RLS policies in schema.sql to actually
-- take effect — superusers bypass Row-Level Security unconditionally.
-- ----------------------------------------------------------------------------
-- DO $$
-- BEGIN
--     IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'threat_detection_app') THEN
--         CREATE ROLE threat_detection_app LOGIN PASSWORD 'CHANGE_ME';
--     END IF;
-- END $$;
-- GRANT CONNECT ON DATABASE threat_detection TO threat_detection_app;
