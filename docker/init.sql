-- Schema for Meta-Evolver's Postgres backend.
--
-- The application creates this itself on connect, so this file exists for the
-- case where it should not: a managed database whose application role has no
-- DDL rights. Apply it once as an admin, then run with
-- PostgresMemoryStore(create_schema=False).
--
-- The vector column is declared at 768 dimensions to match the default
-- embedding width (gemini-embedding-2 truncated via Matryoshka). Change both
-- together -- a column width and an embedding width that disagree means every
-- row silently falls out of the index.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS meta_evolver_memories (
    id                   text NOT NULL,
    namespace            text NOT NULL DEFAULT 'default',
    title                text NOT NULL DEFAULT '',
    scenario             text NOT NULL DEFAULT 'general',
    lesson               text NOT NULL DEFAULT '',
    procedure            text NOT NULL DEFAULT '',
    triggers             jsonb NOT NULL DEFAULT '[]'::jsonb,
    polarity             text NOT NULL DEFAULT 'success',
    source_task_ids      jsonb NOT NULL DEFAULT '[]'::jsonb,
    benchmark            text NOT NULL DEFAULT '',
    -- Source of truth: dimension-agnostic, so a bank holding a mix of remote
    -- and locally-encoded vectors still round-trips.
    embedding_json       jsonb,
    -- The index. Only rows whose width matches get one; the rest are stored
    -- and returned, just invisible to ANN.
    embedding            vector(768),
    uses                 integer NOT NULL DEFAULT 0,
    wins                 integer NOT NULL DEFAULT 0,
    created_generation   integer NOT NULL DEFAULT 0,
    last_used_generation integer NOT NULL DEFAULT 0,
    created_at           timestamptz NOT NULL DEFAULT now(),
    updated_at           timestamptz NOT NULL DEFAULT now(),
    -- Composite on purpose. A memory's id is derived from its text, so two
    -- namespaces can legitimately derive the same one; keyed on id alone, the
    -- second write would update the first namespace's row and disappear from
    -- the bank that created it.
    PRIMARY KEY (namespace, id)
);

CREATE INDEX IF NOT EXISTS meta_evolver_memories_namespace_idx
    ON meta_evolver_memories (namespace);

-- HNSW with cosine ops, matching the normalized unit vectors this project
-- stores and the `1 - (embedding <=> query)` similarity it reads back.
CREATE INDEX IF NOT EXISTS meta_evolver_memories_embedding_idx
    ON meta_evolver_memories USING hnsw (embedding vector_cosine_ops);
