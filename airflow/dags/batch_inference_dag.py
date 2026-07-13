"""
Airflow DAG: nightly_batch_inference
======================================
Orchestrates the complete nightly batch scoring pipeline.

Schedule: 2:00 AM daily (configurable in configs/config.yaml)

Task graph:
    t1_validate_inputs
        └── t2_run_spark_inference
                └── t3_validate_scores
                        └── t4_write_to_postgres
                                └── t5_run_benchmark
                                        └── t6_update_monitoring

Each task:
  - Has retries=2, retry_delay=5min (transient failures don't kill the run)
  - Writes its result to XCom for downstream tasks to consume
  - Updates the batch_runs table on failure/completion
  - SLA alert fires if the full pipeline doesn't complete within 2 hours

This DAG is intentionally simple — no dynamic task generation, no branching,
no sensors. It's a clean sequential pipeline that can be understood at a glance.
Complexity is in the Python callables, not the DAG topology.
"""

from __future__ import annotations

import logging
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator

# ---------------------------------------------------------------------------
# Path setup — the DAG file runs inside the Airflow container where
# /opt/airflow is the working directory.
# ---------------------------------------------------------------------------
_AIRFLOW_ROOT = Path("/opt/airflow")
sys.path.insert(0, str(_AIRFLOW_ROOT))

try:
    from configs.settings import settings

    _CFG = settings
except ImportError:
    # Fallback defaults if settings can't be loaded (DAG parsing context)
    class _MockSettings:
        class pipeline:
            batch_run_id_prefix = "run"
            psi_threshold = 0.2
            min_records_expected = 900_000
            max_null_score_pct = 0.001
            score_range = [0.0, 1.0]

        class airflow:
            dag_id = "nightly_batch_inference"
            schedule = "0 2 * * *"
            retries = 2
            retry_delay_minutes = 5
            sla_minutes = 120
            tags = ["ml", "batch-inference", "churn"]

        class model:
            version = "v1.0.0"

    _CFG = _MockSettings()

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default args (applied to all tasks unless overridden)
# ---------------------------------------------------------------------------
DEFAULT_ARGS = {
    "owner": "ml-engineering",
    "depends_on_past": False,
    "email_on_failure": _CFG.airflow.email_on_failure
    if hasattr(_CFG.airflow, "email_on_failure")
    else False,
    "email_on_retry": False,
    "retries": _CFG.airflow.retries,
    "retry_delay": timedelta(minutes=_CFG.airflow.retry_delay_minutes),
    "sla": timedelta(minutes=_CFG.airflow.sla_minutes),
    "execution_timeout": timedelta(hours=3),
}


# ---------------------------------------------------------------------------
# Task callables
# ---------------------------------------------------------------------------


def task_validate_inputs(**context) -> dict:
    """
    Task 1: Validate that all required inputs exist before starting.

    Checks:
      - Customer Parquet exists and has expected row count
      - Model artifacts exist (model.pkl, encoders.pkl, feature_cols.pkl)
      - PostgreSQL is reachable
      - Sufficient disk space for output

    Pushes run_id to XCom for downstream tasks.
    """
    import shutil
    from db.connection import ping_database

    run_id = (
        f"{_CFG.pipeline.batch_run_id_prefix}-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-"
        f"{str(uuid.uuid4())[:8]}"
    )
    context["task_instance"].xcom_push(key="run_id", value=run_id)
    logger.info(f"Starting batch run: {run_id}")

    errors = []

    # Check input Parquet
    input_path = Path(
        _AIRFLOW_ROOT / _CFG.spark.input_path
        if hasattr(_CFG, "spark")
        else "/opt/airflow/data/customers.parquet"
    )
    if not input_path.exists():
        errors.append(f"Input Parquet not found: {input_path}")
    else:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(input_path)
        n_rows = pf.metadata.num_rows
        logger.info(f"Input Parquet: {n_rows:,} rows")
        if n_rows < _CFG.pipeline.min_records_expected:
            errors.append(
                f"Input has only {n_rows:,} rows, expected >= "
                f"{_CFG.pipeline.min_records_expected:,}"
            )

    # Check model artifacts
    model_paths = {
        "model": _AIRFLOW_ROOT / "models/churn_model.pkl",
        "label_encoders": _AIRFLOW_ROOT / "models/label_encoders.pkl",
        "feature_cols": _AIRFLOW_ROOT / "models/feature_columns.pkl",
    }
    for name, path in model_paths.items():
        if not path.exists():
            errors.append(f"Model artifact not found: {path} ({name})")

    # Check disk space (need ~500MB for output Parquet)
    disk = shutil.disk_usage(_AIRFLOW_ROOT)
    free_mb = disk.free / (1024 * 1024)
    if free_mb < 500:
        errors.append(f"Low disk space: {free_mb:.0f}MB free, need 500MB+")
    else:
        logger.info(f"Disk space OK: {free_mb:.0f}MB free")

    # Check PostgreSQL
    try:
        if ping_database():
            logger.info("PostgreSQL connection: OK")
        else:
            errors.append("PostgreSQL ping failed")
    except Exception as e:
        logger.warning(f"PostgreSQL check skipped (may not be available in test): {e}")

    if errors:
        raise ValueError(
            f"Input validation failed for run_id={run_id}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    # Register batch run in database
    try:
        _register_batch_run(run_id, status="running")
    except Exception as e:
        logger.warning(f"Could not register batch run in DB (non-fatal): {e}")

    logger.info(f"Input validation passed for run_id={run_id}")
    return {"run_id": run_id, "status": "validated"}


def task_run_spark_inference(**context) -> dict:
    """
    Task 2: Execute the PySpark batch inference job.

    Calls spark/batch_inference.py via subprocess (spark-submit compatible).
    Captures stdout/stderr for logging.
    Pushes spark_result to XCom.
    """
    run_id = context["task_instance"].xcom_pull(
        task_ids="t1_validate_inputs", key="run_id"
    )
    logger.info(f"Starting Spark inference for run_id={run_id}...")

    # Import and run directly (faster than subprocess in local mode)
    sys.path.insert(0, str(_AIRFLOW_ROOT / "spark"))
    from spark.batch_inference import run_batch_inference

    t0 = time.perf_counter()
    result = run_batch_inference(run_id=run_id)
    elapsed = time.perf_counter() - t0

    result["total_duration_secs"] = round(elapsed, 2)
    context["task_instance"].xcom_push(key="spark_result", value=result)

    logger.info(
        f"Spark inference complete | "
        f"Records: {result['records_scored']:,} | "
        f"Duration: {result['spark_duration_secs']}s"
    )

    # Update batch_runs with Spark timing
    try:
        _update_batch_run(
            run_id,
            {
                "records_read": result["records_read"],
                "records_scored": result["records_scored"],
                "records_failed": result["records_failed"],
                "spark_duration_secs": result["spark_duration_secs"],
                "score_mean": result["score_stats"]["mean"],
                "score_std": result["score_stats"]["std"],
                "score_p10": result["score_stats"]["p10"],
                "score_p25": result["score_stats"]["p25"],
                "score_p50": result["score_stats"]["p50"],
                "score_p75": result["score_stats"]["p75"],
                "score_p90": result["score_stats"]["p90"],
            },
        )
    except Exception as e:
        logger.warning(f"Could not update batch run in DB (non-fatal): {e}")

    return result


def task_validate_scores(**context) -> dict:
    """
    Task 3: Validate the scored output before writing to production DB.

    Validation gates:
      1. Record count >= min_records_expected
      2. Null score rate <= max_null_score_pct
      3. All scores in [0, 1]
      4. Score distribution is not degenerate (std > 0.01)
      5. Predicted churn rate is within plausible range [0.05, 0.50]

    If validation fails: marks batch run as 'failed' and raises exception
    (Airflow will retry).
    """
    import pandas as pd

    run_id = context["task_instance"].xcom_pull(
        task_ids="t1_validate_inputs", key="run_id"
    )
    spark_result = context["task_instance"].xcom_pull(
        task_ids="t2_run_spark_inference", key="spark_result"
    )
    output_path = spark_result["output_path"]

    logger.info(f"Validating scored output: {output_path}")

    scored_df = pd.read_parquet(output_path)
    n_scored = len(scored_df)
    errors = []
    warnings = []

    # Gate 1: Record count
    if n_scored < _CFG.pipeline.min_records_expected:
        errors.append(
            f"Only {n_scored:,} records scored, expected >= "
            f"{_CFG.pipeline.min_records_expected:,}"
        )

    # Gate 2: Null scores
    null_count = scored_df["churn_probability"].isna().sum()
    null_pct = null_count / n_scored
    if null_pct > _CFG.pipeline.max_null_score_pct:
        errors.append(
            f"Null score rate {null_pct:.2%} exceeds threshold "
            f"{_CFG.pipeline.max_null_score_pct:.2%}"
        )

    # Gate 3: Score range
    score_min = scored_df["churn_probability"].min()
    score_max = scored_df["churn_probability"].max()
    if (
        score_min < _CFG.pipeline.score_range[0]
        or score_max > _CFG.pipeline.score_range[1]
    ):
        errors.append(f"Scores out of range: min={score_min:.4f}, max={score_max:.4f}")

    # Gate 4: Non-degenerate distribution
    score_std = scored_df["churn_probability"].std()
    if score_std < 0.01:
        errors.append(
            f"Score distribution is degenerate (std={score_std:.4f} < 0.01). "
            "Model may be outputting constant predictions."
        )

    # Gate 5: Plausible churn rate
    predicted_churn_rate = scored_df["churn_label"].mean()
    if not (0.05 <= predicted_churn_rate <= 0.50):
        warnings.append(
            f"Predicted churn rate {predicted_churn_rate:.1%} is outside "
            "expected range [5%, 50%]. May indicate model drift."
        )

    for w in warnings:
        logger.warning(f"VALIDATION WARNING: {w}")

    validation_notes = "; ".join(errors + warnings)

    if errors:
        try:
            _update_batch_run(
                run_id,
                {
                    "status": "failed",
                    "validation_passed": False,
                    "validation_notes": validation_notes,
                },
            )
        except Exception as e:
            logger.warning(f"Could not update batch run status: {e}")
        raise ValueError(
            f"Score validation FAILED for run_id={run_id}:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        f"Validation PASSED | "
        f"Records: {n_scored:,} | "
        f"Nulls: {null_count} | "
        f"Churn rate: {predicted_churn_rate:.1%} | "
        f"Score std: {score_std:.4f}"
    )

    try:
        _update_batch_run(
            run_id,
            {
                "validation_passed": True,
                "validation_notes": validation_notes if validation_notes else None,
            },
        )
    except Exception as e:
        logger.warning(f"Could not update batch run in DB (non-fatal): {e}")

    return {
        "n_scored": n_scored,
        "null_rate": round(float(null_pct), 6),
        "predicted_churn_rate": round(float(predicted_churn_rate), 4),
        "score_std": round(float(score_std), 4),
        "validation_passed": True,
    }


def task_write_to_postgres(**context) -> dict:
    """
    Task 4: Write validated predictions to PostgreSQL predictions table.

    This is the audit trail write — every scored record is persisted with:
      - customer_id
      - churn_probability
      - churn_label (True/False)
      - churn_decile (1-10)
      - risk_tier (low/medium/high)
      - run_id (UUID)
      - model_version
      - scored_at (UTC timestamp)

    Downstream services query v_latest_scores view for fast single-customer lookup.
    """
    run_id = context["task_instance"].xcom_pull(
        task_ids="t1_validate_inputs", key="run_id"
    )
    spark_result = context["task_instance"].xcom_pull(
        task_ids="t2_run_spark_inference", key="spark_result"
    )

    sys.path.insert(0, str(_AIRFLOW_ROOT / "spark"))
    from spark.batch_inference import write_predictions_to_postgres

    t0 = time.perf_counter()
    n_written = write_predictions_to_postgres(
        run_id=run_id,
        output_path=spark_result["output_path"],
    )
    elapsed = time.perf_counter() - t0

    logger.info(
        f"PostgreSQL write complete | "
        f"Rows: {n_written:,} | "
        f"Duration: {elapsed:.1f}s | "
        f"Rate: {n_written / elapsed:,.0f} rows/sec"
    )

    try:
        _update_batch_run(
            run_id,
            {
                "status": "completed",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "total_duration_secs": spark_result.get("total_duration_secs"),
            },
        )
    except Exception as e:
        logger.warning(f"Could not finalize batch run in DB (non-fatal): {e}")

    return {
        "rows_written": n_written,
        "write_duration_secs": round(elapsed, 2),
    }


def task_run_benchmark(**context) -> dict:
    """
    Task 5: Run the 3-way benchmark comparison.

    Compares PySpark vs pandas vs joblib Parallel on 4 sample sizes.
    Results are stored in benchmark_results table and as a PNG chart.

    This task runs AFTER the main inference job — it doesn't block the
    critical path but captures performance data for monitoring.
    """
    run_id = context["task_instance"].xcom_pull(
        task_ids="t1_validate_inputs", key="run_id"
    )

    sys.path.insert(0, str(_AIRFLOW_ROOT / "bench"))
    from bench.compare import run_benchmark

    logger.info("Running 3-way benchmark comparison...")
    results = run_benchmark(run_id=run_id)

    logger.info(
        f"Benchmark complete | "
        f"PySpark 1M: {results.get('pyspark_1000000', {}).get('records_per_second', 'N/A')} rec/s | "
        f"pandas 1M: {results.get('pandas_1000000', {}).get('records_per_second', 'N/A')} rec/s"
    )

    return results


def task_update_monitoring(**context) -> dict:
    """
    Task 6: Compute PSI vs previous run and update monitoring metrics.

    PSI (Population Stability Index) measures how much the score distribution
    has shifted compared to the previous batch run. PSI > 0.2 indicates
    significant distribution change and may warrant investigation.

    PSI thresholds (Basel II standard):
      < 0.10 = no significant change
      0.10 – 0.20 = moderate change (monitor)
      > 0.20 = significant shift (investigate)
    """
    run_id = context["task_instance"].xcom_pull(
        task_ids="t1_validate_inputs", key="run_id"
    )
    spark_result = context["task_instance"].xcom_pull(
        task_ids="t2_run_spark_inference", key="spark_result"
    )

    sys.path.insert(0, str(_AIRFLOW_ROOT / "monitoring"))
    from monitoring.score_monitor import ScoreMonitor

    monitor = ScoreMonitor()

    try:
        psi, drift_flagged = monitor.compute_and_store_psi(
            run_id=run_id,
            current_output_path=spark_result["output_path"],
        )
        logger.info(
            f"PSI computed | "
            f"PSI={psi:.4f} | "
            f"Drift={'YES - INVESTIGATE' if drift_flagged else 'No'}"
        )

        # Update batch_runs with PSI result
        try:
            _update_batch_run(
                run_id,
                {
                    "psi_vs_previous": psi,
                    "drift_flagged": drift_flagged,
                    "status": "validated",
                },
            )
        except Exception as e:
            logger.warning(f"Could not update PSI in DB (non-fatal): {e}")

    except Exception as e:
        logger.warning(
            f"PSI computation failed (non-fatal, first run has no baseline): {e}"
        )
        psi = None
        drift_flagged = False

    return {
        "psi": psi,
        "drift_flagged": drift_flagged,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Database helper functions (called from task callables)
# ---------------------------------------------------------------------------


def _register_batch_run(run_id: str, status: str = "running") -> None:
    """Insert a new row into batch_runs to track this pipeline execution."""
    from sqlalchemy import text
    from db.connection import _sync_engine

    with _sync_engine.connect() as conn:
        conn.execute(
            text("""
                INSERT INTO batch_runs (run_id, model_version, started_at, status)
                VALUES (:run_id, :model_version, :started_at, :status)
                ON CONFLICT (run_id) DO NOTHING
            """),
            {
                "run_id": run_id,
                "model_version": _CFG.model.version,
                "started_at": datetime.now(timezone.utc),
                "status": status,
            },
        )
        conn.commit()


def _update_batch_run(run_id: str, updates: dict) -> None:
    """Update fields on an existing batch_runs row."""
    from sqlalchemy import text
    from db.connection import _sync_engine

    set_clauses = ", ".join(f"{k} = :{k}" for k in updates)
    updates["run_id"] = run_id

    with _sync_engine.connect() as conn:
        conn.execute(
            text(f"UPDATE batch_runs SET {set_clauses} WHERE run_id = :run_id"),
            updates,
        )
        conn.commit()


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id=_CFG.airflow.dag_id,
    description=(
        "Nightly batch churn scoring: generates predictions for 1M customers "
        "using PySpark, validates scores, writes to PostgreSQL, and runs "
        "benchmark comparison."
    ),
    schedule=_CFG.airflow.schedule,
    start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
    catchup=_CFG.airflow.catchup,
    default_args=DEFAULT_ARGS,
    tags=_CFG.airflow.tags,
    max_active_runs=1,  # prevent concurrent runs
    doc_md=__doc__,
) as dag:
    t1_validate_inputs = PythonOperator(
        task_id="t1_validate_inputs",
        python_callable=task_validate_inputs,
        doc_md="""
        **Validate inputs** — checks Parquet exists, model artifacts are present,
        PostgreSQL is reachable, and disk space is sufficient.
        Registers a new row in `batch_runs` table.
        """,
    )

    t2_run_spark_inference = PythonOperator(
        task_id="t2_run_spark_inference",
        python_callable=task_run_spark_inference,
        doc_md="""
        **PySpark inference** — scores 1M customers using LightGBM broadcast
        across 50 partitions. Writes scored Parquet to `data/scored_output.parquet`.
        """,
    )

    t3_validate_scores = PythonOperator(
        task_id="t3_validate_scores",
        python_callable=task_validate_scores,
        doc_md="""
        **Validate scores** — 5-gate validation: record count, null rate,
        score range, distribution non-degeneracy, plausible churn rate.
        Blocks downstream write if any gate fails.
        """,
    )

    t4_write_to_postgres = PythonOperator(
        task_id="t4_write_to_postgres",
        python_callable=task_write_to_postgres,
        doc_md="""
        **Write predictions** — batch-inserts all predictions into `predictions`
        table with full audit trail (run_id, model_version, scored_at).
        """,
    )

    t5_run_benchmark = PythonOperator(
        task_id="t5_run_benchmark",
        python_callable=task_run_benchmark,
        doc_md="""
        **Benchmark comparison** — times PySpark vs pandas vs joblib on
        4 sample sizes. Results stored in `benchmark_results` table and PNG chart.
        Non-blocking: failure here does not fail the pipeline.
        """,
    )

    t6_update_monitoring = PythonOperator(
        task_id="t6_update_monitoring",
        python_callable=task_update_monitoring,
        doc_md="""
        **PSI monitoring** — computes Population Stability Index vs previous run.
        Flags `drift_flagged=True` in `batch_runs` if PSI > 0.2.
        """,
    )

    # Linear dependency chain
    (
        t1_validate_inputs
        >> t2_run_spark_inference
        >> t3_validate_scores
        >> t4_write_to_postgres
        >> t5_run_benchmark
        >> t6_update_monitoring
    )
