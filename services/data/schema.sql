-- Data Service schema for PostgreSQL with pgvector.
--
-- Book: "Designing AI Systems"
--   Listing 5.18: PostgreSQL schema for vector storage with pgvector
--   Listing 5.22: Adding full-text search to the chunks table

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR(255) PRIMARY KEY,
    document_id VARCHAR(255) NOT NULL,
    index_name VARCHAR(255) NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding vector,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_index ON chunks(index_name);
CREATE INDEX IF NOT EXISTS idx_chunks_metadata ON chunks USING gin (metadata);

-- IVFFlat index for fast similarity search.
-- Best created after loading representative data, not on an empty table.
-- CREATE INDEX idx_chunks_embedding ON chunks
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- Listing 5.22: Full-text search column (auto-maintained by PostgreSQL)
ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (to_tsvector('english', chunk_text)) STORED;

CREATE INDEX IF NOT EXISTS idx_chunks_search ON chunks USING gin (search_vector);

-- Index metadata table for managing indexes.
CREATE TABLE IF NOT EXISTS data_indexes (
    name VARCHAR(255) PRIMARY KEY,
    config JSONB NOT NULL,
    owner VARCHAR(255) DEFAULT '',
    document_count INT DEFAULT 0,
    total_chunks INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_ingested_at TIMESTAMP
);

-- Document metadata table.
CREATE TABLE IF NOT EXISTS documents (
    document_id VARCHAR(255) NOT NULL,
    index_name VARCHAR(255) NOT NULL,
    filename VARCHAR(1024) NOT NULL,
    chunk_count INT DEFAULT 0,
    page_count INT,
    word_count INT,
    custom_metadata JSONB DEFAULT '{}',
    ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (index_name, document_id)
);
