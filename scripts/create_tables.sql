CREATE TABLE IF NOT EXISTS long_term_memory (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL DEFAULT '',
    content TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    embedding DOUBLE PRECISION[],
    category TEXT NOT NULL DEFAULT 'general',
    tags TEXT[] NOT NULL DEFAULT '{}',
    slot_hint TEXT NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_accessed TIMESTAMP NOT NULL DEFAULT NOW(),
    base_importance REAL,
    decay_anchor TIMESTAMP,
    fact JSONB
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id TEXT PRIMARY KEY,
    username TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '新对话',
    message_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS session_messages (
    id SERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
    msg_type TEXT NOT NULL,
    content TEXT NOT NULL,
    rag_trace JSONB,
    created_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS user_preferences (
    id SERIAL PRIMARY KEY,
    username TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    UNIQUE(username, key)
);

CREATE INDEX IF NOT EXISTS idx_ltm_username ON long_term_memory(username);
CREATE INDEX IF NOT EXISTS idx_ltm_category ON long_term_memory(category);
CREATE INDEX IF NOT EXISTS idx_sessions_username ON chat_sessions(username);
CREATE INDEX IF NOT EXISTS idx_msgs_session ON session_messages(session_id);
CREATE INDEX IF NOT EXISTS idx_prefs_username ON user_preferences(username);
