-- Phase 3: PostgreSQL is the canonical store for messages and controlled memory.

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    session_id TEXT NOT NULL,
    turn_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    source_type TEXT NOT NULL DEFAULT 'voice',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(session_id, turn_id, role)
);

ALTER TABLE chat_sessions
    ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;
ALTER TABLE session_messages
    ADD COLUMN IF NOT EXISTS turn_id TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS uq_session_messages_canonical_turn
    ON session_messages(session_id, turn_id, msg_type)
    WHERE turn_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_type TEXT NOT NULL CHECK (session_type IN ('voice', 'text')),
    session_id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ,
    PRIMARY KEY(session_type, session_id),
    UNIQUE(session_type, session_id, user_id)
);

INSERT INTO conversation_sessions(
    session_type, session_id, user_id, created_at, ended_at
)
SELECT 'voice', session_id, user_id, created_at, ended_at
FROM voice_sessions
ON CONFLICT (session_type, session_id) DO UPDATE SET
    ended_at = COALESCE(
        EXCLUDED.ended_at, conversation_sessions.ended_at
    )
WHERE conversation_sessions.user_id = EXCLUDED.user_id;

INSERT INTO conversation_sessions(
    session_type, session_id, user_id, created_at, ended_at
)
SELECT 'text', sessions.session_id, users.user_id,
       sessions.created_at, sessions.ended_at
FROM chat_sessions AS sessions
JOIN users ON users.username = sessions.username
ON CONFLICT (session_type, session_id) DO UPDATE SET
    ended_at = COALESCE(
        EXCLUDED.ended_at, conversation_sessions.ended_at
    )
WHERE conversation_sessions.user_id = EXCLUDED.user_id;

-- Earlier Phase 3 development revisions constrained this unified table to
-- voice_sessions.  Remove those constraints idempotently so text and voice use
-- the same canonical message store; ownership remains enforced by user_id.
ALTER TABLE conversation_messages
    DROP CONSTRAINT IF EXISTS conversation_messages_session_id_fkey;
ALTER TABLE conversation_messages
    DROP CONSTRAINT IF EXISTS conversation_messages_session_id_turn_id_fkey;

CREATE TABLE IF NOT EXISTS session_summaries (
    summary_id UUID PRIMARY KEY,
    session_type TEXT NOT NULL CHECK (session_type IN ('voice', 'text')),
    session_id TEXT NOT NULL,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    summary_version INTEGER NOT NULL CHECK (summary_version > 0),
    content TEXT NOT NULL,
    structured_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_digest TEXT NOT NULL,
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_session_summaries_unified_version
        UNIQUE(session_type, session_id, summary_version),
    CONSTRAINT session_summaries_unified_session_fkey
        FOREIGN KEY(session_type, session_id, user_id)
        REFERENCES conversation_sessions(session_type, session_id, user_id)
        ON DELETE CASCADE
);

ALTER TABLE session_summaries
    ADD COLUMN IF NOT EXISTS session_type TEXT;
UPDATE session_summaries AS summaries
SET session_type = CASE
    WHEN EXISTS (
        SELECT 1 FROM conversation_sessions AS sessions
        WHERE sessions.session_type = 'voice'
          AND sessions.session_id = summaries.session_id
          AND sessions.user_id = summaries.user_id
    ) THEN 'voice'
    WHEN EXISTS (
        SELECT 1 FROM conversation_sessions AS sessions
        WHERE sessions.session_type = 'text'
          AND sessions.session_id = summaries.session_id
          AND sessions.user_id = summaries.user_id
    ) THEN 'text'
END
WHERE session_type IS NULL;
ALTER TABLE session_summaries
    ALTER COLUMN session_type SET NOT NULL;
ALTER TABLE session_summaries
    DROP CONSTRAINT IF EXISTS session_summaries_session_id_fkey;
ALTER TABLE session_summaries
    DROP CONSTRAINT IF EXISTS session_summaries_session_id_summary_version_key;
ALTER TABLE session_summaries
    DROP CONSTRAINT IF EXISTS session_summaries_session_type_check;
ALTER TABLE session_summaries
    ADD CONSTRAINT session_summaries_session_type_check
    CHECK (session_type IN ('voice', 'text'));
ALTER TABLE session_summaries
    DROP CONSTRAINT IF EXISTS session_summaries_unified_session_fkey;
ALTER TABLE session_summaries
    ADD CONSTRAINT session_summaries_unified_session_fkey
    FOREIGN KEY(session_type, session_id, user_id)
    REFERENCES conversation_sessions(session_type, session_id, user_id)
    ON DELETE CASCADE;
CREATE UNIQUE INDEX IF NOT EXISTS uq_session_summaries_unified_version
    ON session_summaries(session_type, session_id, summary_version);

CREATE TABLE IF NOT EXISTS medical_fact_memories (
    memory_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    memory_type TEXT NOT NULL,
    content TEXT NOT NULL,
    structured_value JSONB NOT NULL DEFAULT '{}'::jsonb,
    status TEXT NOT NULL CHECK (status IN ('proposed', 'confirmed', 'superseded', 'rejected')),
    source_type TEXT NOT NULL CHECK (source_type IN ('user_message', 'personal_document', 'user_correction')),
    source_session_type TEXT CHECK (source_session_type IN ('voice', 'text')),
    source_session_id TEXT,
    source_turn_id TEXT,
    source_document_id TEXT,
    valid_from TIMESTAMPTZ,
    valid_to TIMESTAMPTZ,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0.0 CHECK (confidence >= 0 AND confidence <= 1),
    supersedes_memory_id UUID REFERENCES medical_fact_memories(memory_id) ON DELETE RESTRICT,
    candidate_key TEXT NOT NULL,
    deleted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(user_id, candidate_key),
    CONSTRAINT medical_fact_memories_unified_session_fkey
        FOREIGN KEY(source_session_type, source_session_id, user_id)
        REFERENCES conversation_sessions(session_type, session_id, user_id)
        ON DELETE RESTRICT
);

ALTER TABLE medical_fact_memories
    ADD COLUMN IF NOT EXISTS source_session_type TEXT;
UPDATE medical_fact_memories AS memories
SET source_session_type = CASE
    WHEN EXISTS (
        SELECT 1 FROM conversation_sessions AS sessions
        WHERE sessions.session_type = 'voice'
          AND sessions.session_id = memories.source_session_id
          AND sessions.user_id = memories.user_id
    ) THEN 'voice'
    WHEN EXISTS (
        SELECT 1 FROM conversation_sessions AS sessions
        WHERE sessions.session_type = 'text'
          AND sessions.session_id = memories.source_session_id
          AND sessions.user_id = memories.user_id
    ) THEN 'text'
END
WHERE source_session_id IS NOT NULL AND source_session_type IS NULL;
ALTER TABLE medical_fact_memories
    DROP CONSTRAINT IF EXISTS medical_fact_memories_source_session_id_fkey;
ALTER TABLE medical_fact_memories
    DROP CONSTRAINT IF EXISTS medical_fact_memories_source_session_type_check;
ALTER TABLE medical_fact_memories
    ADD CONSTRAINT medical_fact_memories_source_session_type_check
    CHECK (source_session_type IS NULL OR source_session_type IN ('voice', 'text'));
ALTER TABLE medical_fact_memories
    DROP CONSTRAINT IF EXISTS medical_fact_memories_source_session_pair_check;
ALTER TABLE medical_fact_memories
    ADD CONSTRAINT medical_fact_memories_source_session_pair_check
    CHECK (
        (source_session_id IS NULL AND source_session_type IS NULL)
        OR (source_session_id IS NOT NULL AND source_session_type IS NOT NULL)
    );
ALTER TABLE medical_fact_memories
    DROP CONSTRAINT IF EXISTS medical_fact_memories_unified_session_fkey;
ALTER TABLE medical_fact_memories
    ADD CONSTRAINT medical_fact_memories_unified_session_fkey
    FOREIGN KEY(source_session_type, source_session_id, user_id)
    REFERENCES conversation_sessions(session_type, session_id, user_id)
    ON DELETE RESTRICT;

-- Only server-side document processing may populate this registry. Worker
-- evidence fields are never sufficient to confirm a medical fact.
CREATE TABLE IF NOT EXISTS trusted_personal_document_contents (
    verification_id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    content_sha256 CHAR(64) NOT NULL,
    status TEXT NOT NULL DEFAULT 'trusted'
        CHECK (status IN ('trusted', 'revoked')),
    verified_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ,
    UNIQUE(user_id, document_id, content_sha256)
);

CREATE INDEX IF NOT EXISTS idx_conversation_messages_session_created
    ON conversation_messages(session_id, created_at, role);
CREATE INDEX IF NOT EXISTS idx_session_summaries_user_created
    ON session_summaries(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_medical_fact_memories_user_status
    ON medical_fact_memories(user_id, status, created_at DESC)
    WHERE deleted_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_medical_fact_memories_supersedes
    ON medical_fact_memories(supersedes_memory_id)
    WHERE supersedes_memory_id IS NOT NULL;
