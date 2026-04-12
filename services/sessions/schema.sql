-- Session Service database schema
-- Run against a PostgreSQL database before starting with postgres storage.
--
-- Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
--   Listing 4.9:  sessions table
--   Listing 4.10: messages table

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR(255) PRIMARY KEY,
    user_id    VARCHAR(255) NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

CREATE TABLE IF NOT EXISTS messages (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(255) NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role        VARCHAR(50)  NOT NULL,
    content     TEXT,
    tool_calls  JSONB,
    tool_call_id VARCHAR(255),
    name        VARCHAR(255),
    timestamp   TIMESTAMP NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);

CREATE TABLE IF NOT EXISTS memories (
    user_id    VARCHAR(255) NOT NULL,
    key        VARCHAR(255) NOT NULL,
    value      JSONB NOT NULL,
    session_id VARCHAR(255) NOT NULL DEFAULT '',
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, key, session_id)
);

CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id);
