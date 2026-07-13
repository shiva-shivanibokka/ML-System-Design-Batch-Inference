"""
PySpark Batch Inference Job
============================
Scores 1,000,000 customer records for churn probability using LightGBM.

Architecture:
    1. Read customers.parquet into a Spark DataFrame
    2. Repartition to N partitions (default 50) for parallel processing
    3. Broadcast the trained model + encoders to all executors
    4. mapPartitions() — each partition runs LightGBM inference independently
       using pandas_udf-style row batching (no Python UDF overhead)
    5. Write scored results back to Parquet
    6. Write predictions to PostgreSQL (via JDBC or pandas batch inserts)
    7. Update batch_runs table with job metadata

Key pattern — model broadcasting:
    The LightGBM model (~1MB) is serialized with joblib and broadcast via
    SparkContext.broadcast(). Each executor deserializes it once per partition,
    not once per row. This is the production pattern used at Airbnb/LinkedIn
    for nightly scoring jobs.

Usage:
    # Local mode (all cores):
    python spark/batch_inference.py --run-id run-20240101-020000

    # Submit to standalone cluster:
    spark-submit --master spark://host:7077 spark/batch_inference.py

    # Via Airflow BashOperator (inside Docker):
    spark-submit /opt/airflow/spark/batch_inference.py --run-id ${run_id}
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path resolution — works when called from project root or Airflow container
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from configs.settings import settings  # noqa: E402  (after sys.path setup above)
from features import predict_scores  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [Spark] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Spark partition inference function
# ---------------------------------------------------------------------------


def _make_partition_predictor(bc_model, bc_encoders, bc_features):
    """
    Factory that builds the mapInPandas closure.

    It closes over the Spark BROADCAST VARIABLES (bc_*), not their raw values.
    That distinction is the whole point of broadcasting: `.value` is read INSIDE
    predict_partition, i.e. on the executor, so Spark ships the model to each
    executor once and caches it — instead of serialising the full model into
    every task closure. (The previous version passed `broadcast.value` on the
    driver, which silently defeated the broadcast and shipped the model per task.)
    Each executor deserialises the model once per partition, not once per row.
    """

    def predict_partition(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        import io
        import joblib

        # Read broadcast values ON THE EXECUTOR (this is what makes it a broadcast).
        model = joblib.load(io.BytesIO(bc_model.value))
        encoders = joblib.load(io.BytesIO(bc_encoders.value))
        feature_cols = bc_features.value

        for batch_df in iterator:
            if batch_df.empty:
                yield batch_df
                continue
            # Feature engineering + inference + output derivation — all shared with
            # training and the pandas scorer via features.py (no skew possible).
            yield predict_scores(model, batch_df, encoders, feature_cols)

    return predict_partition


# ---------------------------------------------------------------------------
# Main job
# ---------------------------------------------------------------------------


def run_batch_inference(run_id: str) -> dict:
    """
    Execute the full PySpark batch inference job.

    Parameters
    ----------
    run_id : Unique identifier for this batch run (e.g. "run-20240101-020000")

    Returns
    -------
    dict with job metadata: records_scored, duration_secs, score_stats, etc.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        DoubleType,
        BooleanType,
        IntegerType,
        StringType,
        StructField,
        StructType,
    )

    cfg_spark = settings.spark
    cfg_model = settings.model

    # --- Verify artifacts exist ---
    for path, name in [
        (cfg_model.path, "model"),
        (cfg_model.label_encoders_path, "label encoders"),
        (cfg_model.feature_columns_path, "feature columns"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}\nRun `python models/train.py` first."
            )

    # --- Load artifacts (on driver) ---
    logger.info("Loading model artifacts on driver...")
    import io

    model = joblib.load(cfg_model.path)
    encoders = joblib.load(cfg_model.label_encoders_path)
    feature_cols = joblib.load(cfg_model.feature_columns_path)

    # Serialize to bytes for broadcasting
    model_buf = io.BytesIO()
    joblib.dump(model, model_buf)
    model_bytes = model_buf.getvalue()
    encoders_buf = io.BytesIO()
    joblib.dump(encoders, encoders_buf)
    encoders_bytes = encoders_buf.getvalue()

    logger.info(
        f"Model artifacts loaded | "
        f"Model size: {len(model_bytes) / 1024:.1f}KB | "
        f"Features: {len(feature_cols)}"
    )

    # --- Create SparkSession ---
    logger.info(f"Initialising SparkSession (master={cfg_spark.master})...")
    spark = (
        SparkSession.builder.appName(f"{cfg_spark.app_name}_{run_id}")
        .master(cfg_spark.master)
        .config("spark.executor.memory", cfg_spark.executor_memory)
        .config("spark.driver.memory", cfg_spark.driver_memory)
        # Kryo serializer is faster than Java default for ML workloads
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # Reduce shuffle partitions for local mode
        .config("spark.sql.shuffle.partitions", str(cfg_spark.n_partitions))
        # Arrow enables fast Spark ↔ pandas conversion in mapInPandas
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel(cfg_spark.log_level)

    # --- Broadcast model artifacts to all executors ---
    logger.info("Broadcasting model artifacts to executors...")
    broadcast_model = spark.sparkContext.broadcast(model_bytes)
    broadcast_encoders = spark.sparkContext.broadcast(encoders_bytes)
    broadcast_features = spark.sparkContext.broadcast(feature_cols)

    # --- Read input data ---
    input_path = cfg_spark.input_path
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Input Parquet not found: {input_path}\n"
            "Run `python data/generate_data.py` first."
        )

    logger.info(f"Reading input Parquet: {input_path}")
    t_read = time.perf_counter()
    df = spark.read.parquet(input_path)
    n_input = df.count()
    read_elapsed = time.perf_counter() - t_read
    logger.info(f"Read {n_input:,} rows in {read_elapsed:.1f}s")

    # --- Repartition for parallel inference ---
    # This is the key operation: splits the dataset into N equal partitions.
    # Each partition is processed independently on a separate thread/core.
    # n_partitions = 50 means 50 × 20K-row chunks for 1M records.
    logger.info(f"Repartitioning to {cfg_spark.n_partitions} partitions...")
    df = df.repartition(cfg_spark.n_partitions)

    # --- Define output schema ---
    output_schema = StructType(
        [
            StructField("customer_id", StringType(), nullable=False),
            StructField("churn_probability", DoubleType(), nullable=False),
            StructField("churn_label", BooleanType(), nullable=False),
            StructField("churn_decile", IntegerType(), nullable=False),
            StructField("risk_tier", StringType(), nullable=False),
        ]
    )

    # --- Run distributed inference using mapInPandas ---
    # mapInPandas passes each partition as a pandas DataFrame iterator.
    # The predict_partition closure captures the broadcast values.
    # This is equivalent to Pandas UDF (SCALAR_ITER) but without Arrow schema
    # limitations — we can return multiple columns directly.
    logger.info(
        f"Starting distributed inference across {cfg_spark.n_partitions} partitions..."
    )
    t_infer = time.perf_counter()

    predict_fn = _make_partition_predictor(
        broadcast_model,
        broadcast_encoders,
        broadcast_features,
    )

    scored_df = df.mapInPandas(predict_fn, schema=output_schema)

    # --- Write scored output to Parquet ---
    output_path = cfg_spark.output_path
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing scored output to {output_path}...")
    (
        scored_df.coalesce(1)  # single output file for easy downstream consumption
        .write.mode("overwrite")
        .parquet(output_path)
    )

    # Trigger execution (Spark is lazy — nothing runs until here)
    n_scored = spark.read.parquet(output_path).count()
    infer_elapsed = time.perf_counter() - t_infer

    logger.info(
        f"Inference complete | "
        f"Scored: {n_scored:,} | "
        f"Duration: {infer_elapsed:.1f}s | "
        f"Throughput: {n_scored / infer_elapsed:,.0f} records/sec"
    )

    # --- Compute score statistics (for audit trail) ---
    logger.info("Computing score statistics...")
    scored_pdf = spark.read.parquet(output_path).select("churn_probability").toPandas()
    probs = scored_pdf["churn_probability"].values

    score_stats = {
        "mean": float(round(probs.mean(), 4)),
        "std": float(round(probs.std(), 4)),
        "p10": float(round(np.percentile(probs, 10), 4)),
        "p25": float(round(np.percentile(probs, 25), 4)),
        "p50": float(round(np.percentile(probs, 50), 4)),
        "p75": float(round(np.percentile(probs, 75), 4)),
        "p90": float(round(np.percentile(probs, 90), 4)),
    }

    # --- Unpersist broadcasts to free executor memory ---
    broadcast_model.unpersist()
    broadcast_encoders.unpersist()
    broadcast_features.unpersist()

    spark.stop()
    logger.info("SparkSession stopped.")

    return {
        "run_id": run_id,
        "model_version": cfg_model.version,
        "records_read": n_input,
        "records_scored": n_scored,
        "records_failed": n_input - n_scored,
        "spark_duration_secs": round(infer_elapsed, 2),
        "score_stats": score_stats,
        "output_path": output_path,
    }


def write_predictions_to_postgres(run_id: str, output_path: str) -> int:
    """
    Batch-insert scored predictions from Parquet into PostgreSQL.
    Uses chunked inserts (50K rows at a time) to avoid memory spikes.

    Returns total rows inserted.
    """
    from db.connection import _sync_engine

    logger.info(f"Writing predictions to PostgreSQL (run_id={run_id})...")

    scored_df = pd.read_parquet(output_path)
    scored_df["run_id"] = run_id
    scored_df["model_version"] = settings.model.version
    scored_df["scored_at"] = datetime.now(timezone.utc)
    scored_df["churn_probability"] = scored_df["churn_probability"].astype(float)
    scored_df["churn_label"] = scored_df["churn_label"].astype(bool)
    scored_df["churn_decile"] = scored_df["churn_decile"].astype(int)

    chunk_size = 50_000
    n_inserted = 0
    n_chunks = (len(scored_df) + chunk_size - 1) // chunk_size

    with _sync_engine.connect() as conn:
        for i, chunk in enumerate(range(0, len(scored_df), chunk_size)):
            batch = scored_df.iloc[chunk : chunk + chunk_size]
            batch.to_sql(
                name="predictions",
                con=conn,
                if_exists="append",
                index=False,
                method="multi",
                chunksize=1000,
            )
            n_inserted += len(batch)
            if (i + 1) % 5 == 0 or (i + 1) == n_chunks:
                logger.info(
                    f"  Inserted {n_inserted:,}/{len(scored_df):,} rows "
                    f"({n_inserted / len(scored_df):.0%})"
                )
        conn.commit()

    logger.info(f"Predictions written to PostgreSQL: {n_inserted:,} rows")
    return n_inserted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PySpark batch inference job — scores 1M customers for churn"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}",
        help="Unique batch run ID (auto-generated if not provided)",
    )
    parser.add_argument(
        "--skip-postgres",
        action="store_true",
        help="Skip writing to PostgreSQL (useful for local testing)",
    )
    args = parser.parse_args()

    run_id = args.run_id
    logger.info(f"Starting batch inference | run_id={run_id}")

    try:
        result = run_batch_inference(run_id=run_id)

        logger.info(
            f"Batch inference complete | "
            f"Scored: {result['records_scored']:,} | "
            f"Duration: {result['spark_duration_secs']}s"
        )
        logger.info(f"Score stats: {result['score_stats']}")

        if not args.skip_postgres:
            write_predictions_to_postgres(run_id, result["output_path"])

    except Exception as e:
        logger.exception(f"Batch inference FAILED for run_id={run_id}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
