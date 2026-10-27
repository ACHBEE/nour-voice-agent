-- Nour / Clinique El Amal: database schema.
-- Loaded automatically by the postgres container on first start
-- (mounted into /docker-entrypoint-initdb.d). Safe to re-run: every statement is idempotent.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Knowledge base
-- ---------------------------------------------------------------------------

-- One row per source file in data/clinic_docs/.
CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    doc_id       TEXT        NOT NULL UNIQUE,          -- stable id from the doc's front matter
    title        TEXT        NOT NULL,
    lang         TEXT        NOT NULL CHECK (lang IN ('ar', 'fr', 'en')),
    dialect      TEXT,                                  -- e.g. 'darja' for Algerian Arabic
    category     TEXT        NOT NULL,
    source_path  TEXT        NOT NULL,
    content_hash TEXT        NOT NULL,                  -- re-ingest only when the file changed
    ingested_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Maps a document language to a Postgres text-search config.
-- Falls back to 'simple' (no stemming) when the server has no stemmer for that language.
CREATE OR REPLACE FUNCTION nour_ts_config(lang TEXT) RETURNS regconfig
LANGUAGE sql STABLE AS $$
    SELECT COALESCE(
        (SELECT c.oid::regconfig
           FROM pg_ts_config c
          WHERE c.cfgname = CASE lang
                                WHEN 'fr' THEN 'french'
                                WHEN 'en' THEN 'english'
                                WHEN 'ar' THEN 'arabic'
                            END),
        'simple'::regconfig)
$$;

-- Retrieval units. Each chunk carries both a dense embedding (semantic, cross-lingual)
-- and a tsvector (exact terms: names, "CNAS", prices) for hybrid search.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    chunk_index INT         NOT NULL,
    section     TEXT,                                   -- markdown heading the chunk came from
    content     TEXT        NOT NULL,
    lang        TEXT        NOT NULL CHECK (lang IN ('ar', 'fr', 'en')),
    token_count INT,
    -- intfloat/multilingual-e5-small -> 384 dims. Changing model = new dimension + re-ingest.
    embedding   vector(384) NOT NULL,
    ts_config   regconfig   NOT NULL DEFAULT 'simple',  -- set by ingestion via nour_ts_config(lang)
    tsv         tsvector GENERATED ALWAYS AS (to_tsvector(ts_config, content)) STORED,
    UNIQUE (document_id, chunk_index)
);

-- ANN index for cosine similarity. HNSW over IVFFlat: no training step, good recall
-- at small scale, and it can be built on an empty table.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

-- Full-text index for the keyword half of hybrid search.
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING gin (tsv);

CREATE INDEX IF NOT EXISTS chunks_lang_idx ON chunks (lang);

-- ---------------------------------------------------------------------------
-- Operations: bookings and human callbacks
-- ---------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS bookings (
    id               BIGSERIAL PRIMARY KEY,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    patient_name     TEXT        NOT NULL CHECK (char_length(patient_name) >= 2),
    phone            TEXT        NOT NULL CHECK (phone ~ '^\+?[0-9]{9,13}$'),
    department       TEXT        NOT NULL CHECK (department IN
                         ('general_medicine', 'pediatrics', 'cardiology', 'gynecology', 'dentistry')),
    appointment_date DATE        NOT NULL,
    appointment_time TIME        NOT NULL,
    language         TEXT        NOT NULL DEFAULT 'fr' CHECK (language IN ('ar', 'fr', 'en')),
    status           TEXT        NOT NULL DEFAULT 'pending'
                         CHECK (status IN ('pending', 'confirmed', 'cancelled')),
    room             TEXT                               -- LiveKit room / call id, for tracing
);

-- No double-booking of the same department slot (cancelled bookings free the slot).
CREATE UNIQUE INDEX IF NOT EXISTS bookings_slot_unique
    ON bookings (department, appointment_date, appointment_time)
    WHERE status <> 'cancelled';

CREATE TABLE IF NOT EXISTS callbacks (
    id         BIGSERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    phone      TEXT        NOT NULL CHECK (phone ~ '^\+?[0-9]{9,13}$'),
    reason     TEXT        NOT NULL,
    language   TEXT        NOT NULL DEFAULT 'fr' CHECK (language IN ('ar', 'fr', 'en')),
    status     TEXT        NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'done')),
    room       TEXT
);
