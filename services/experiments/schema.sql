-- Experimentation Service database schema.
--
-- Book: "Designing AI Systems" (https://www.manning.com/books/designing-ai-systems)
--   Listing 7.14: Experimentation Service gRPC contract
--   Listing 7.15: ExperimentTarget / EvaluationSummary
--
-- Mirrors the in-memory store's shape; the servicer is store-agnostic.

CREATE TABLE IF NOT EXISTS experiment_targets (
    name               VARCHAR(128) NOT NULL,
    version            INTEGER NOT NULL,
    type               VARCHAR(32) NOT NULL DEFAULT 'PROMPT',
    author             VARCHAR(128) NOT NULL DEFAULT '',
    change_description TEXT NOT NULL DEFAULT '',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status             VARCHAR(16) NOT NULL DEFAULT 'DRAFT',
    evaluation_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (name, version)
);

CREATE INDEX IF NOT EXISTS idx_experiment_targets_name   ON experiment_targets(name);
CREATE INDEX IF NOT EXISTS idx_experiment_targets_status ON experiment_targets(status);

CREATE TABLE IF NOT EXISTS datasets (
    name       VARCHAR(128) PRIMARY KEY,
    metadata   JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS test_cases (
    id              VARCHAR(64) PRIMARY KEY,
    dataset_name    VARCHAR(128) NOT NULL REFERENCES datasets(name) ON DELETE CASCADE,
    input_query     TEXT NOT NULL DEFAULT '',
    ideal_response  TEXT NOT NULL DEFAULT '',
    key_elements    JSONB NOT NULL DEFAULT '[]'::jsonb,
    tags            JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_trace_id VARCHAR(64) NOT NULL DEFAULT '',
    needs_review    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_test_cases_dataset ON test_cases(dataset_name);

CREATE TABLE IF NOT EXISTS evaluations (
    evaluation_id VARCHAR(64) PRIMARY KEY,
    dataset_name  VARCHAR(128) NOT NULL,
    target_ids    JSONB NOT NULL DEFAULT '[]'::jsonb,
    metrics       JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status        VARCHAR(16) NOT NULL DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    evaluation_id     VARCHAR(64) PRIMARY KEY REFERENCES evaluations(evaluation_id) ON DELETE CASCADE,
    dataset_name      VARCHAR(128) NOT NULL,
    target_results    JSONB NOT NULL DEFAULT '[]'::jsonb,
    per_case_results  JSONB NOT NULL DEFAULT '[]'::jsonb
);

CREATE TABLE IF NOT EXISTS scoring_rules (
    name        VARCHAR(128) PRIMARY KEY,
    workflow_id VARCHAR(128) NOT NULL DEFAULT '',
    sample_rate DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    scorers     JSONB NOT NULL DEFAULT '[]'::jsonb,
    alerts      JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_scoring_rules_workflow ON scoring_rules(workflow_id);

CREATE TABLE IF NOT EXISTS experiments (
    name                 VARCHAR(128) PRIMARY KEY,
    workflow_id          VARCHAR(128) NOT NULL DEFAULT '',
    variants             JSONB NOT NULL DEFAULT '[]'::jsonb,
    success_metrics      JSONB NOT NULL DEFAULT '[]'::jsonb,
    minimum_sample_size  INTEGER NOT NULL DEFAULT 100,
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status               VARCHAR(16) NOT NULL DEFAULT 'draft'
);

CREATE TABLE IF NOT EXISTS variant_assignments (
    assignment_id   VARCHAR(64) PRIMARY KEY,
    experiment_name VARCHAR(128) NOT NULL,
    assignment_key  VARCHAR(128) NOT NULL,
    variant         JSONB NOT NULL DEFAULT '{}'::jsonb,
    assigned_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_name, assignment_key)
);

CREATE INDEX IF NOT EXISTS idx_variant_assignments_experiment
    ON variant_assignments(experiment_name);

CREATE TABLE IF NOT EXISTS outcomes (
    id              BIGSERIAL PRIMARY KEY,
    experiment_name VARCHAR(128) NOT NULL,
    assignment_id   VARCHAR(64) NOT NULL,
    variant_name    VARCHAR(128) NOT NULL DEFAULT '',
    outcomes        JSONB NOT NULL DEFAULT '{}'::jsonb,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_outcomes_experiment ON outcomes(experiment_name);

CREATE TABLE IF NOT EXISTS annotation_queues (
    name          VARCHAR(128) PRIMARY KEY,
    workflow_id   VARCHAR(128) NOT NULL DEFAULT '',
    rubric        JSONB NOT NULL DEFAULT '[]'::jsonb,
    routing_rules JSONB NOT NULL DEFAULT '[]'::jsonb,
    reviewers     JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS annotation_items (
    item_id     VARCHAR(64) PRIMARY KEY,
    queue_name  VARCHAR(128) NOT NULL REFERENCES annotation_queues(name) ON DELETE CASCADE,
    trace_id    VARCHAR(64) NOT NULL DEFAULT '',
    status      VARCHAR(16) NOT NULL DEFAULT 'pending',
    assigned_to VARCHAR(128) NOT NULL DEFAULT '',
    added_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_annotation_items_queue ON annotation_items(queue_name, status);
