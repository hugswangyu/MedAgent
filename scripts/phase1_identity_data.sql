CREATE TABLE IF NOT EXISTS users (
    user_id UUID PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS voice_sessions (
    session_id TEXT PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    knowledge_base_id TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'created',
    room_name TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS voice_turns (
    turn_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES voice_sessions(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    turn_index INTEGER,
    user_text TEXT,
    assistant_text TEXT,
    safety_result JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(session_id, turn_id)
);

CREATE TABLE IF NOT EXISTS evidence (
    evidence_id TEXT PRIMARY KEY,
    turn_id TEXT NOT NULL,
    session_id TEXT NOT NULL REFERENCES voice_sessions(session_id) ON DELETE CASCADE,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE RESTRICT,
    source_type TEXT NOT NULL,
    source_id TEXT NOT NULL DEFAULT '',
    document_id TEXT NOT NULL DEFAULT '',
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    FOREIGN KEY(session_id, turn_id)
        REFERENCES voice_turns(session_id, turn_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS audit_events (
    audit_id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(user_id) ON DELETE SET NULL,
    session_id TEXT,
    turn_id TEXT,
    actor_type TEXT NOT NULL,
    action TEXT NOT NULL,
    outcome TEXT NOT NULL,
    request_id TEXT,
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS worker_request_nonces (
    token_jti UUID NOT NULL,
    nonce UUID NOT NULL,
    user_id UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY(token_jti, nonce)
);

CREATE INDEX IF NOT EXISTS idx_voice_sessions_user_created
    ON voice_sessions(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_voice_turns_session_created
    ON voice_turns(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_evidence_turn ON evidence(turn_id);
CREATE INDEX IF NOT EXISTS idx_audit_user_created
    ON audit_events(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_worker_nonces_expires
    ON worker_request_nonces(expires_at);
