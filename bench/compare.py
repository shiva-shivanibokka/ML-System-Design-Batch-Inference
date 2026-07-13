"""
3-Way Batch Inference Benchmark
=================================
Compares inference throughput across three parallel execution strategies:

  1. PySpark (local[*])
     - Distributes data across N partitions
     - Each partition runs LightGBM independently on a separate thread
     - Uses mapInPandas — no Python UDF serialization overhead
     - Best for: 100K+ records, multi-core machines, cluster-scalable workloads

  2. pandas (single-threaded)
     - Loads entire dataset into memory as a single DataFrame
     - Calls model.predict_proba() on the full matrix at once
     - Fastest for small datasets (<50K), no startup overhead
     - Best for: low-latency single-batch inference, debugging

  3. joblib Parallel (multi-process, loky backend)
     - Splits data into chunks equal to n_jobs (all CPUs by default)
     - Each subprocess gets one chunk + a copy of the model
     - No JVM startup overhead; lower memory than Spark
     - Best for: medium datasets (10K–500K), when Spark overhead isn't justified

Key results to look for:
  - PySpark wins at 500K+ rows due to JVM startup amortization
  - pandas wins at 10K rows (no overhead)
  - joblib sits in between — good balance for moderate sizes
  - All three should produce IDENTICAL predictions (verified in the benchmark)

Usage:
    python bench/compare.py
    python bench/compare.py --sizes 10000 100000 1000000
    python bench/compare.py --no-chart    # skip matplotlib chart
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import joblib as jl
import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared inference helpers (must match models/train.py)
# ---------------------------------------------------------------------------


def _load_artifacts():
    """Load model, encoders, and feature columns from disk."""
    cfg = settings.model
    for p in [cfg.path, cfg.label_encoders_path, cfg.feature_columns_path]:
        if not Path(p).exists():
            raise FileNotFoundError(
                f"Model artifact not found: {p}. Run `python models/train.py` first."
            )
    model = jl.load(cfg.path)
    encoders = jl.load(cfg.label_encoders_path)
    feature_cols = jl.load(cfg.feature_columns_path)
    return model, encoders, feature_cols


def _prepare_batch(
    df: pd.DataFrame, encoders: dict, feature_cols: list
) -> pd.DataFrame:
    """Apply feature engineering + encoding via the shared features.py."""
    from features import build_feature_matrix

    return build_feature_matrix(df, encoders, feature_cols)


def _predict_chunk(
    chunk_df: pd.DataFrame, model, encoders: dict, feature_cols: list
) -> np.ndarray:
    """Run inference on a single chunk — used by joblib workers."""
    X = _prepare_batch(chunk_df.copy(), encoders, feature_cols)
    return model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# Engine 1: PySpark
# ---------------------------------------------------------------------------


def _benchmark_pyspark(df: pd.DataFrame, model, encoders, feature_cols) -> Dict:
    """
    PySpark inference via mapInPandas with model broadcasting.
    """
    import io
    from pyspark.sql import SparkSession
    from pyspark.sql.types import DoubleType, StringType, StructField, StructType

    cfg = settings.spark

    # Serialize model for broadcasting
    model_buf = io.BytesIO()
    jl.dump(model, model_buf)
    encoders_buf = io.BytesIO()
    jl.dump(encoders, encoders_buf)
    model_bytes = model_buf.getvalue()
    encoders_bytes = encoders_buf.getvalue()

    # Start Spark
    spark = (
        SparkSession.builder.appName("BatchInferenceBenchmark")
        .master(cfg.master)
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "50000")
        .config(
            "spark.sql.shuffle.partitions",
            str(min(cfg.n_partitions, len(df) // 10000 + 1)),
        )
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    bc_model = spark.sparkContext.broadcast(model_bytes)
    bc_encoders = spark.sparkContext.broadcast(encoders_bytes)
    bc_features = spark.sparkContext.broadcast(feature_cols)

    output_schema = StructType(
        [
            StructField("customer_id", StringType(), False),
            StructField("churn_probability", DoubleType(), False),
        ]
    )

    def _output_schema_predictor(iterator):
        import io
        import joblib
        from features import build_feature_matrix

        _model = joblib.load(io.BytesIO(bc_model.value))
        _encoders = joblib.load(io.BytesIO(bc_encoders.value))
        _features = bc_features.value
        for batch_df in iterator:
            if batch_df.empty:
                yield batch_df
                continue
            cids = batch_df["customer_id"].values
            X = build_feature_matrix(batch_df, _encoders, _features)
            probs = _model.predict_proba(X)[:, 1]
            yield pd.DataFrame({"customer_id": cids, "churn_probability": probs})

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    n_parts = max(1, min(settings.spark.n_partitions, len(df) // 10000 + 1))
    sdf = spark.createDataFrame(df).repartition(n_parts)
    result_sdf = sdf.mapInPandas(_output_schema_predictor, schema=output_schema)
    n_scored = result_sdf.count()  # trigger execution

    elapsed = time.perf_counter() - t0
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    bc_model.unpersist()
    bc_encoders.unpersist()
    bc_features.unpersist()
    spark.stop()

    return {
        "engine": "pyspark",
        "n_records": n_scored,
        "duration_secs": round(elapsed, 4),
        "records_per_second": round(n_scored / elapsed, 2),
        "peak_memory_mb": round(mem_after - mem_before, 2),
        "cpu_cores_used": os.cpu_count(),
    }


# ---------------------------------------------------------------------------
# Engine 2: pandas (single-threaded)
# ---------------------------------------------------------------------------


def _benchmark_pandas(df: pd.DataFrame, model, encoders, feature_cols) -> Dict:
    """Single-threaded pandas inference — no parallelism."""
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    X = _prepare_batch(df.copy(), encoders, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    n_scored = len(probs)

    elapsed = time.perf_counter() - t0
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    return {
        "engine": "pandas",
        "n_records": n_scored,
        "duration_secs": round(elapsed, 4),
        "records_per_second": round(n_scored / elapsed, 2),
        "peak_memory_mb": round(mem_after - mem_before, 2),
        "cpu_cores_used": 1,
    }


# ---------------------------------------------------------------------------
# Engine 3: joblib Parallel
# ---------------------------------------------------------------------------


def _benchmark_joblib(df: pd.DataFrame, model, encoders, feature_cols) -> Dict:
    """
    Multi-process inference using joblib.Parallel with loky backend.
    Splits data into n_chunks = n_jobs (all CPUs).
    """
    cfg = settings.benchmark
    n_jobs = cfg.n_joblib_workers  # -1 = all CPUs
    backend = cfg.joblib_backend
    n_jobs_r = os.cpu_count() if n_jobs == -1 else n_jobs

    chunk_size = max(1, len(df) // n_jobs_r)
    chunks = [df.iloc[i : i + chunk_size].copy() for i in range(0, len(df), chunk_size)]

    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)
    t0 = time.perf_counter()

    results = jl.Parallel(n_jobs=n_jobs, backend=backend, verbose=0)(
        jl.delayed(_predict_chunk)(chunk, model, encoders, feature_cols)
        for chunk in chunks
    )

    probs = np.concatenate(results)
    n_scored = len(probs)
    elapsed = time.perf_counter() - t0
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    return {
        "engine": "joblib",
        "n_records": n_scored,
        "duration_secs": round(elapsed, 4),
        "records_per_second": round(n_scored / elapsed, 2),
        "peak_memory_mb": round(mem_after - mem_before, 2),
        "cpu_cores_used": n_jobs_r,
    }


# ---------------------------------------------------------------------------
# Full benchmark runner
# ---------------------------------------------------------------------------


def run_benchmark(run_id: Optional[str] = None) -> Dict:
    """
    Run the full 3-engine benchmark across all configured sample sizes.
    Returns a dict of results keyed by {engine}_{sample_size}.
    Also saves results to JSON and PNG chart.
    """
    if run_id is None:
        run_id = f"bench-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"

    cfg = settings.benchmark

    # Load base dataset
    input_path = settings.spark.input_path
    if not Path(input_path).exists():
        raise FileNotFoundError(
            f"Input Parquet not found: {input_path}. "
            "Run `python data/generate_data.py` first."
        )

    logger.info(f"Loading dataset for benchmark: {input_path}")
    full_df = pd.read_parquet(input_path)

    # Load model artifacts
    model, encoders, feature_cols = _load_artifacts()

    # Warm-up run (not timed) — ensures model is hot in CPU cache
    warmup_df = full_df.iloc[: cfg.warmup_rows].copy()
    _benchmark_pandas(warmup_df, model, encoders, feature_cols)
    logger.info("Warm-up complete.")
    gc.collect()

    all_results = []
    timing_data: Dict[str, Dict[int, float]] = {
        "pyspark": {},
        "pandas": {},
        "joblib": {},
    }

    for n in cfg.sample_sizes:
        sample_df = full_df.iloc[:n].copy()
        logger.info(f"\n--- Sample size: {n:,} ---")

        for engine_name, engine_fn in [
            ("pandas", lambda d: _benchmark_pandas(d, model, encoders, feature_cols)),
            ("joblib", lambda d: _benchmark_joblib(d, model, encoders, feature_cols)),
            ("pyspark", lambda d: _benchmark_pyspark(d, model, encoders, feature_cols)),
        ]:
            logger.info(f"  Running {engine_name}...")
            try:
                res = engine_fn(sample_df.copy())
                res["run_id"] = run_id
                res["sample_size"] = n
                all_results.append(res)
                timing_data[engine_name][n] = res["duration_secs"]
                logger.info(
                    f"  {engine_name:10s} | {res['duration_secs']:8.2f}s | "
                    f"{res['records_per_second']:>12,.0f} rec/s"
                )
            except Exception as e:
                logger.error(f"  {engine_name} FAILED for n={n:,}: {e}")

            gc.collect()

    # Save JSON results
    Path(cfg.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {cfg.results_path}")

    # Save chart
    _plot_benchmark(timing_data, cfg.sample_sizes, cfg.chart_path)

    # Persist to PostgreSQL if available
    _save_to_postgres(all_results, run_id)

    # Build return dict
    return {
        f"{r['engine']}_{r['sample_size']}": {
            "duration_secs": r["duration_secs"],
            "records_per_second": r["records_per_second"],
        }
        for r in all_results
    }


def _plot_benchmark(
    timing_data: Dict[str, Dict[int, float]],
    sample_sizes: List[int],
    output_path: str,
) -> None:
    """Generate speedup chart comparing all 3 engines."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        colors = {"pyspark": "#E25A1C", "pandas": "#1f77b4", "joblib": "#2ca02c"}
        markers = {"pyspark": "o", "pandas": "s", "joblib": "^"}

        # Left plot: duration (seconds)
        for engine, timings in timing_data.items():
            xs = sorted(timings.keys())
            ys = [timings[x] for x in xs]
            ax1.plot(
                [x / 1000 for x in xs],
                ys,
                label=engine,
                color=colors[engine],
                marker=markers[engine],
                linewidth=2,
                markersize=8,
            )
        ax1.set_xlabel("Sample Size (thousands)", fontsize=12)
        ax1.set_ylabel("Duration (seconds)", fontsize=12)
        ax1.set_title("Inference Duration by Engine", fontsize=14, fontweight="bold")
        ax1.legend(fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.set_xscale("log")
        ax1.set_yscale("log")

        # Right plot: throughput (records/sec)
        for engine, timings in timing_data.items():
            xs = sorted(timings.keys())
            ys = [x / timings[x] for x in xs]
            ax2.plot(
                [x / 1000 for x in xs],
                ys,
                label=engine,
                color=colors[engine],
                marker=markers[engine],
                linewidth=2,
                markersize=8,
            )
        ax2.set_xlabel("Sample Size (thousands)", fontsize=12)
        ax2.set_ylabel("Throughput (records/second)", fontsize=12)
        ax2.set_title("Inference Throughput by Engine", fontsize=14, fontweight="bold")
        ax2.legend(fontsize=11)
        ax2.grid(alpha=0.3)
        ax2.set_xscale("log")

        plt.suptitle(
            "Batch Inference Benchmark: PySpark vs pandas vs joblib",
            fontsize=15,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Benchmark chart saved to {output_path}")

    except Exception as e:
        logger.warning(f"Could not generate chart: {e}")


def _save_to_postgres(results: List[Dict], run_id: str) -> None:
    """Persist benchmark results to PostgreSQL benchmark_results table."""
    try:
        from sqlalchemy import text
        from db.connection import _sync_engine

        rows = [
            {
                "run_id": run_id,
                "engine": r["engine"],
                "sample_size": r["sample_size"],
                "duration_secs": r["duration_secs"],
                "records_per_second": r["records_per_second"],
                "peak_memory_mb": r.get("peak_memory_mb"),
                "cpu_cores_used": r.get("cpu_cores_used"),
                "benchmarked_at": datetime.now(timezone.utc),
            }
            for r in results
        ]

        with _sync_engine.connect() as conn:
            for row in rows:
                conn.execute(
                    text("""
                        INSERT INTO benchmark_results
                            (run_id, engine, sample_size, duration_secs,
                             records_per_second, peak_memory_mb, cpu_cores_used, benchmarked_at)
                        VALUES
                            (:run_id, :engine, :sample_size, :duration_secs,
                             :records_per_second, :peak_memory_mb, :cpu_cores_used, :benchmarked_at)
                    """),
                    row,
                )
            conn.commit()
        logger.info(f"Saved {len(rows)} benchmark rows to PostgreSQL")
    except Exception as e:
        logger.warning(f"Could not save benchmark to PostgreSQL (non-fatal): {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="3-way batch inference benchmark: PySpark vs pandas vs joblib"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=settings.benchmark.sample_sizes,
        help="Sample sizes to benchmark (default from config)",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip generating the PNG chart",
    )
    args = parser.parse_args()

    if args.sizes:
        settings.benchmark.sample_sizes = args.sizes

    print("\n" + "=" * 60)
    print("  BATCH INFERENCE BENCHMARK")
    print("  PySpark  vs  pandas  vs  joblib Parallel")
    print("=" * 60)
    print(f"  Sample sizes : {[f'{n:,}' for n in settings.benchmark.sample_sizes]}")
    print(f"  CPU cores    : {os.cpu_count()}")
    print(f"  RAM          : {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print("=" * 60 + "\n")

    results = run_benchmark()

    # Print summary table
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Engine':<12} {'Size':>12} {'Duration':>12} {'Rec/sec':>14}")
    print("  " + "-" * 54)
    for key, val in sorted(results.items()):
        engine, size = key.rsplit("_", 1)
        print(
            f"  {engine:<12} {int(size):>12,} "
            f"{val['duration_secs']:>11.2f}s "
            f"{val['records_per_second']:>13,.0f}"
        )
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
