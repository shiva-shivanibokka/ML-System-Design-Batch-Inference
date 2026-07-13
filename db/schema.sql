-- =============================================================================
-- Batch Inference Database Schema
-- =============================================================================
-- PostgreSQL 16
-- Run once at startup: psql -U batch_user -d batch_inference -f schema.sql
-- =============================================================================

-- ---------------------------------------------------------------------------
-- Extensions
-- ---------------------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- customers
-- Source table — the customer-level view of the KKBox dataset, one row per
-- member (msno). This mirrors data/customers.parquet, which is what the batch
-- pipeline actually reads. In production this would be populated by upstream
-- ETL; here data/build_dataset.py produces the equivalent Parquet.
-- Columns are the raw KKBox features (see configs/config.yaml); the model's
-- derived ratio/flag features are computed at inference time in features.py.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS customers (
    customer_id             VARCHAR(64)     PRIMARY KEY,   -- KKBox msno (hashed, ~44 chars)
    -- Demographics (members_v3)
    city                    INTEGER,
    bd                      INTEGER,                        -- age; dirty in KKBox, nullable
    gender                  VARCHAR(16),
    registered_via          INTEGER,
    registration_days       INTEGER,                        -- days registered before cutoff
    -- Subscription/payment behaviour (aggregated from transactions_v2)
    n_transactions          INTEGER         NOT NULL DEFAULT 0,
    total_paid              NUMERIC(14, 2)  NOT NULL DEFAULT 0,
    avg_plan_price          NUMERIC(10, 2),
    avg_plan_days           NUMERIC(8, 2),
    total_discount          NUMERIC(14, 2)  NOT NULL DEFAULT 0,
    n_auto_renew            INTEGER         NOT NULL DEFAULT 0,
    n_cancels               INTEGER         NOT NULL DEFAULT 0,
    membership_tenure_days  INTEGER,
    days_to_expire          INTEGER,
    payment_method_id       INTEGER,
    last_is_auto_renew      SMALLINT,
    last_is_cancel          SMALLINT,
    -- Ground-truth label (KKBox train_v2)
    is_churn                SMALLINT,
    -- Metadata
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- ---------------------------------------------------------------------------
-- batch_runs
-- One row per pipeline execution. Tracks job metadata, timing, and health.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS batch_runs (
    run_id                  VARCHAR(64)     PRIMARY KEY,
    model_version           VARCHAR(32)     NOT NULL,
    started_at              TIMESTAMPTZ     NOT NULL,
    completed_at            TIMESTAMPTZ,
    status                  VARCHAR(16)     NOT NULL DEFAULT 'running'
                                            CHECK (status IN ('running', 'completed', 'failed', 'validated')),
    -- Volume
    records_read            INTEGER,
    records_scored          INTEGER,
    records_failed          INTEGER         DEFAULT 0,
    -- Timing (seconds)
    spark_duration_secs     NUMERIC(10, 2),
    total_duration_secs     NUMERIC(10, 2),
    -- Score statistics (populated after scoring)
    score_mean              NUMERIC(6, 4),
    score_std               NUMERIC(6, 4),
    score_p10               NUMERIC(6, 4),
    score_p25               NUMERIC(6, 4),
    score_p50               NUMERIC(6, 4),
    score_p75               NUMERIC(6, 4),
    score_p90               NUMERIC(6, 4),
    -- Drift monitoring
    psi_vs_previous         NUMERIC(8, 4),  -- Population Stability Index vs last run
    drift_flagged           BOOLEAN         DEFAULT FALSE,
    -- Validation
    validation_passed       BOOLEAN,
    validation_notes        TEXT,
    -- Benchmark results (serialised JSON)
    benchmark_results_json  TEXT,
    -- Free-form metadata
    notes                   TEXT,
    created_at              TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_batch_runs_started_at ON batch_runs (started_at DESC);
CREATE INDEX IF NOT EXISTS idx_batch_runs_status ON batch_runs (status);

-- ---------------------------------------------------------------------------
-- predictions
-- One row per customer per batch run.
-- This is the audit table — full history of every score ever produced.
-- Downstream services read the LATEST score per customer_id.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
    id                      BIGSERIAL       PRIMARY KEY,
    customer_id             VARCHAR(64)     NOT NULL,       -- KKBox msno (~44 chars)
    run_id                  VARCHAR(64)     NOT NULL REFERENCES batch_runs(run_id),
    model_version           VARCHAR(32)     NOT NULL,
    churn_probability       NUMERIC(6, 4)   NOT NULL CHECK (churn_probability BETWEEN 0 AND 1),
    churn_label             BOOLEAN         NOT NULL,   -- TRUE if probability >= 0.5
    churn_decile            SMALLINT        NOT NULL CHECK (churn_decile BETWEEN 1 AND 10),
    -- Risk tier derived from probability
    risk_tier               VARCHAR(8)      NOT NULL CHECK (risk_tier IN ('low', 'medium', 'high')),
    scored_at               TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

-- Critical index: API lookup of latest score for a customer is O(log n)
CREATE INDEX IF NOT EXISTS idx_predictions_customer_run
    ON predictions (customer_id, run_id);

-- Index for "latest score per customer" query pattern
CREATE INDEX IF NOT EXISTS idx_predictions_customer_scored
    ON predictions (customer_id, scored_at DESC);

-- Index for batch run analytics (e.g. score distribution for a specific run)
CREATE INDEX IF NOT EXISTS idx_predictions_run_id ON predictions (run_id);

-- Partial index for high-risk customers (most queried in downstream campaigns)
CREATE INDEX IF NOT EXISTS idx_predictions_high_risk
    ON predictions (customer_id, scored_at DESC)
    WHERE risk_tier = 'high';

-- ---------------------------------------------------------------------------
-- benchmark_results
-- Stores timing results from each 3-way benchmark comparison run.
-- Used by the Gradio dashboard's Benchmark tab.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS benchmark_results (
    id                      SERIAL          PRIMARY KEY,
    run_id                  VARCHAR(64)     NOT NULL,
    engine                  VARCHAR(32)     NOT NULL CHECK (engine IN ('pyspark', 'pandas', 'joblib')),
    sample_size             INTEGER         NOT NULL,
    duration_secs           NUMERIC(10, 4)  NOT NULL,
    records_per_second      NUMERIC(12, 2)  NOT NULL,
    peak_memory_mb          NUMERIC(10, 2),
    cpu_cores_used          SMALLINT,
    benchmarked_at          TIMESTAMPTZ     NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_benchmark_run_id ON benchmark_results (run_id);
CREATE INDEX IF NOT EXISTS idx_benchmark_engine ON benchmark_results (engine, sample_size);

-- ---------------------------------------------------------------------------
-- Convenience views
-- ---------------------------------------------------------------------------

-- Latest score per customer (used by the /score/{customer_id} API endpoint)
CREATE OR REPLACE VIEW v_latest_scores AS
SELECT DISTINCT ON (customer_id)
    customer_id,
    run_id,
    model_version,
    churn_probability,
    churn_label,
    churn_decile,
    risk_tier,
    scored_at
FROM predictions
ORDER BY customer_id, scored_at DESC;

-- Batch run summary for dashboard
CREATE OR REPLACE VIEW v_batch_run_summary AS
SELECT
    r.run_id,
    r.model_version,
    r.started_at,
    r.completed_at,
    r.status,
    r.records_scored,
    r.spark_duration_secs,
    r.total_duration_secs,
    r.score_mean,
    r.score_p50,
    r.psi_vs_previous,
    r.drift_flagged,
    r.validation_passed,
    -- Derived
    ROUND(r.records_scored::NUMERIC / NULLIF(r.spark_duration_secs, 0), 0) AS records_per_second,
    EXTRACT(EPOCH FROM (r.completed_at - r.started_at)) AS wall_clock_secs
FROM batch_runs r
ORDER BY r.started_at DESC;

-- Score distribution histogram per run (10 bins)
CREATE OR REPLACE VIEW v_score_histogram AS
SELECT
    run_id,
    width_bucket(churn_probability, 0, 1, 10) AS bin,
    ROUND((width_bucket(churn_probability, 0, 1, 10) - 1) * 0.1, 1) AS bin_lower,
    ROUND(width_bucket(churn_probability, 0, 1, 10) * 0.1, 1)       AS bin_upper,
    COUNT(*)                                                           AS count,
    ROUND(COUNT(*)::NUMERIC / SUM(COUNT(*)) OVER (PARTITION BY run_id), 4) AS fraction
FROM predictions
GROUP BY run_id, bin
ORDER BY run_id, bin;
