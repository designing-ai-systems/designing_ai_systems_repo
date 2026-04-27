-- Workflow Service Postgres schema.
--
-- Book: "Designing AI Systems" — Listing 8.9.
--
-- Ships as documentation only in commit 1 of the chapter-8 plan; the
-- in-memory `WorkflowRegistry` and `JobStore` are the production
-- backends used by the book demo. A `PostgresJobStore` /
-- `PostgresWorkflowRegistry` against this schema is the natural
-- follow-up — see `chapters/book_discrepancies_chapter8.md` discrepancy #2.

CREATE TABLE IF NOT EXISTS workflows (
    workflow_id        VARCHAR PRIMARY KEY,
    name               VARCHAR UNIQUE NOT NULL,
    api_path           VARCHAR NOT NULL,
    container_image    VARCHAR,
    response_mode      VARCHAR NOT NULL DEFAULT 'sync',
    spec_json          JSONB NOT NULL,
    version            INTEGER NOT NULL DEFAULT 1,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS workflows_api_path_idx ON workflows (api_path);

CREATE TABLE IF NOT EXISTS workflow_deployments (
    deployment_id        VARCHAR PRIMARY KEY,
    workflow_id          VARCHAR NOT NULL REFERENCES workflows (workflow_id) ON DELETE CASCADE,
    version              INTEGER NOT NULL,
    status               VARCHAR NOT NULL,
    current_replicas     INTEGER NOT NULL DEFAULT 0,
    desired_replicas     INTEGER NOT NULL DEFAULT 0,
    healthy_endpoints    JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at           TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS workflow_deployments_workflow_idx
    ON workflow_deployments (workflow_id);

-- Listing 8.9
CREATE TABLE IF NOT EXISTS workflow_jobs (
    job_id              VARCHAR PRIMARY KEY,
    workflow_id         VARCHAR NOT NULL,
    status              VARCHAR NOT NULL,
    progress_message    TEXT NOT NULL DEFAULT '',
    input_json          TEXT NOT NULL DEFAULT '',
    result_json         TEXT NOT NULL DEFAULT '',
    error               TEXT NOT NULL DEFAULT '',
    checkpoint_json     TEXT NOT NULL DEFAULT '',
    assigned_endpoint   VARCHAR NOT NULL DEFAULT '',
    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS workflow_jobs_status_idx ON workflow_jobs (status);
CREATE INDEX IF NOT EXISTS workflow_jobs_workflow_idx ON workflow_jobs (workflow_id);
