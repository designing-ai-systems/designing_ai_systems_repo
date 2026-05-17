-- Observability Service database schema.
--
-- Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
--   Listing 7.2:  log events
--   Listing 7.5:  spans, generations, traces
--   Listing 7.11: scores
--
-- Single Postgres database with separate tables for each primitive.
-- The chapter's "Log Store / Time-Series DB / Trace + Generation Store /
-- Score Store" four-store model is collapsed here for runnable book
-- demos (see chapters/book_discrepancies_chapter7.md).

CREATE TABLE IF NOT EXISTS spans (
    span_id           VARCHAR(64) PRIMARY KEY,
    trace_id          VARCHAR(64) NOT NULL,
    parent_span_id    VARCHAR(64) NOT NULL DEFAULT '',
    service           VARCHAR(64) NOT NULL,
    operation         VARCHAR(256) NOT NULL,
    start_time        TIMESTAMPTZ NOT NULL,
    end_time          TIMESTAMPTZ,
    status            VARCHAR(16) NOT NULL DEFAULT 'OK',
    error_message     TEXT NOT NULL DEFAULT '',
    attributes        JSONB NOT NULL DEFAULT '{}'::jsonb,
    numeric_attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    events            JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_spans_trace_id   ON spans(trace_id);
CREATE INDEX IF NOT EXISTS idx_spans_service    ON spans(service);
CREATE INDEX IF NOT EXISTS idx_spans_start_time ON spans(start_time);

CREATE TABLE IF NOT EXISTS generations (
    span_id           VARCHAR(64) PRIMARY KEY,
    trace_id          VARCHAR(64) NOT NULL,
    parent_span_id    VARCHAR(64) NOT NULL DEFAULT '',
    service           VARCHAR(64) NOT NULL,
    operation         VARCHAR(256) NOT NULL,
    start_time        TIMESTAMPTZ NOT NULL,
    end_time          TIMESTAMPTZ,
    status            VARCHAR(16) NOT NULL DEFAULT 'OK',
    error_message     TEXT NOT NULL DEFAULT '',
    attributes        JSONB NOT NULL DEFAULT '{}'::jsonb,
    numeric_attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    -- LLM-specific fields (Listing 7.5).
    model             VARCHAR(128) NOT NULL,
    provider          VARCHAR(64) NOT NULL DEFAULT '',
    requested_model   VARCHAR(128) NOT NULL DEFAULT '',
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd          DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    cache_hit         BOOLEAN NOT NULL DEFAULT FALSE,
    fallback_used     BOOLEAN NOT NULL DEFAULT FALSE,
    time_to_first_token_ms DOUBLE PRECISION NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_generations_trace_id   ON generations(trace_id);
CREATE INDEX IF NOT EXISTS idx_generations_model      ON generations(model);
CREATE INDEX IF NOT EXISTS idx_generations_provider   ON generations(provider);
CREATE INDEX IF NOT EXISTS idx_generations_start_time ON generations(start_time);

CREATE TABLE IF NOT EXISTS logs (
    event_id          VARCHAR(64) PRIMARY KEY,
    trace_id          VARCHAR(64) NOT NULL DEFAULT '',
    span_id           VARCHAR(64) NOT NULL DEFAULT '',
    timestamp         TIMESTAMPTZ NOT NULL,
    service           VARCHAR(64) NOT NULL DEFAULT '',
    severity          VARCHAR(16) NOT NULL DEFAULT 'INFO',
    event_type        VARCHAR(64) NOT NULL DEFAULT '',
    message           TEXT NOT NULL DEFAULT '',
    attributes        JSONB NOT NULL DEFAULT '{}'::jsonb,
    numeric_attributes JSONB NOT NULL DEFAULT '{}'::jsonb,
    workflow_id       VARCHAR(64) NOT NULL DEFAULT '',
    user_id           VARCHAR(64) NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_logs_trace_id  ON logs(trace_id);
CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp);

CREATE TABLE IF NOT EXISTS metrics (
    id        BIGSERIAL PRIMARY KEY,
    name      VARCHAR(128) NOT NULL,
    type      VARCHAR(16) NOT NULL DEFAULT 'COUNTER',
    value     DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    labels    JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON metrics(name, timestamp);

CREATE TABLE IF NOT EXISTS scores (
    score_id       VARCHAR(64) PRIMARY KEY,
    trace_id       VARCHAR(64) NOT NULL DEFAULT '',
    span_id        VARCHAR(64) NOT NULL DEFAULT '',
    generation_id  VARCHAR(64) NOT NULL DEFAULT '',
    name           VARCHAR(128) NOT NULL,
    -- Score.value is a Union[float, str, bool]; we keep three columns plus
    -- a discriminator so SQL filters can target the right kind cheaply.
    value_kind     VARCHAR(16) NOT NULL DEFAULT 'numeric',
    numeric_value  DOUBLE PRECISION,
    string_value   TEXT,
    boolean_value  BOOLEAN,
    source         VARCHAR(32) NOT NULL DEFAULT 'AUTOMATED',
    comment        TEXT NOT NULL DEFAULT '',
    metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
    timestamp      TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_scores_trace_id ON scores(trace_id);
CREATE INDEX IF NOT EXISTS idx_scores_name     ON scores(name);

CREATE TABLE IF NOT EXISTS budget_alerts (
    name                  VARCHAR(128) PRIMARY KEY,
    scope_type            VARCHAR(32) NOT NULL DEFAULT 'team',
    scope_value           VARCHAR(128) NOT NULL DEFAULT '',
    limit_usd             DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    period                VARCHAR(32) NOT NULL DEFAULT 'monthly',
    thresholds            JSONB NOT NULL DEFAULT '[0.7, 0.9, 1.0]'::jsonb,
    notification_channels JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
