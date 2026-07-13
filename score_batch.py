"""
Pandas batch scorer — the deployed / CI scoring path.
=====================================================
Scores the customer table with the trained LightGBM model using plain pandas
(no Spark / JVM). This is the right-sized engine for the volumes that run in
GitHub Actions and serverless; spark/batch_inference.py is the same logic for
cluster-scale runs. The benchmark (bench/compare.py) quantifies where each wins.

Writes the scored Parquet and — unless --skip-postgres — the predictions and
batch_runs audit rows that the API and dashboard read.

Usage:
    python score_batch.py                       # score data/customers.parquet → Neon
    python score_batch.py --skip-postgres       # local, Parquet only
    python score_batch.py --input data/sample.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from configs.settings import settings
from features import predict_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [score] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def score(run_id: str, input_path: str | None = None, skip_postgres: bool = False) -> dict:
    cfg_model = settings.model
    input_path = input_path or settings.spark.input_path

    for path, name in [
        (cfg_model.path, "model"),
        (cfg_model.label_encoders_path, "label encoders"),
        (cfg_model.feature_columns_path, "feature columns"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{name} not found at {path}. Run models/train.py first.")
    if not Path(input_path).exists():
        raise FileNotFoundError(f"Input not found: {input_path}. Run data/build_dataset.py first.")

    model = joblib.load(cfg_model.path)
    encoders = joblib.load(cfg_model.label_encoders_path)
    feature_cols = joblib.load(cfg_model.feature_columns_path)

    df = pd.read_parquet(input_path)
    logger.info("Scoring %s rows with model %s", f"{len(df):,}", cfg_model.version)

    t0 = time.perf_counter()
    scored = predict_scores(model, df, encoders, feature_cols)
    dur = time.perf_counter() - t0

    _validate(scored, n_read=len(df))  # reject a bad batch before it reaches the DB

    out = settings.spark.output_path
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out, index=False)

    probs = scored["churn_probability"].to_numpy()
    stats = {
        "mean": float(round(probs.mean(), 4)),
        "std": float(round(probs.std(), 4)),
        "p50": float(round(np.percentile(probs, 50), 4)),
        "p90": float(round(np.percentile(probs, 90), 4)),
    }
    logger.info(
        "Scored %s rows in %.1fs (%.0f rec/s) | churn rate %.1f%%",
        f"{len(scored):,}", dur, len(scored) / dur, 100 * scored["churn_label"].mean(),
    )

    if not skip_postgres:
        _write_to_postgres(run_id, out, stats, dur, len(df), len(scored))

    return {"run_id": run_id, "records_scored": len(scored), "duration_secs": round(dur, 2), "score_stats": stats}


def _validate(scored: pd.DataFrame, n_read: int) -> None:
    """
    The same gates as the Airflow path, but scale-relative so they hold on a
    20K sample and a 1M full run alike. Raises on any hard failure (the nightly
    job then fails loudly instead of writing a bad batch).
    """
    cfg = settings.pipeline
    prob = scored["churn_probability"]
    errors = []

    if len(scored) < 0.9 * n_read:
        errors.append(f"only {len(scored):,}/{n_read:,} rows scored (>10% dropped)")
    null_rate = prob.isna().mean()
    if null_rate > cfg.max_null_score_pct:
        errors.append(f"null score rate {null_rate:.2%} > {cfg.max_null_score_pct:.2%}")
    if prob.min() < cfg.score_range[0] or prob.max() > cfg.score_range[1]:
        errors.append(f"scores out of range [{prob.min():.3f}, {prob.max():.3f}]")
    if prob.std() < 0.01:
        errors.append(f"degenerate distribution (std={prob.std():.4f}) — constant predictions?")

    churn_rate = scored["churn_label"].mean()
    if not (0.05 <= churn_rate <= 0.60):
        logger.warning("Predicted churn rate %.1f%% is outside [5%%, 60%%] — possible drift", 100 * churn_rate)

    if errors:
        raise ValueError("Validation FAILED: " + "; ".join(errors))
    logger.info("Validation passed (%s rows, churn %.1f%%, std %.3f)", f"{len(scored):,}", 100 * churn_rate, prob.std())


def _write_to_postgres(run_id, output_path, stats, dur, n_read, n_scored) -> None:
    """Register the run, write predictions, finalise batch_runs — the audit trail."""
    from sqlalchemy import text
    from db.connection import _sync_engine
    from spark.batch_inference import write_predictions_to_postgres

    with _sync_engine.connect() as conn:
        conn.execute(
            text(
                "INSERT INTO batch_runs (run_id, model_version, started_at, status) "
                "VALUES (:r, :m, :s, 'running') ON CONFLICT (run_id) DO NOTHING"
            ),
            {"r": run_id, "m": settings.model.version, "s": datetime.now(timezone.utc)},
        )
        conn.commit()

    write_predictions_to_postgres(run_id, output_path)

    # PSI drift vs the previous run (same monitor the Airflow path uses), so the
    # dashboard's drift chart populates on the deployed path too.
    psi, drift = None, False
    try:
        from monitoring.score_monitor import ScoreMonitor

        psi, drift = ScoreMonitor().compute_and_store_psi(run_id, output_path)
    except Exception as e:  # first run has no baseline — non-fatal
        logger.warning("PSI not computed: %s", e)

    with _sync_engine.connect() as conn:
        conn.execute(
            text(
                "UPDATE batch_runs SET status='validated', validation_passed=TRUE, "
                "completed_at=:c, records_read=:rr, records_scored=:rs, "
                "spark_duration_secs=:d, score_mean=:mean, score_p50=:p50, score_p90=:p90, "
                "psi_vs_previous=:psi, drift_flagged=:drift "
                "WHERE run_id=:r"
            ),
            {
                "c": datetime.now(timezone.utc), "rr": n_read, "rs": n_scored, "d": round(dur, 2),
                "mean": stats["mean"], "p50": stats["p50"], "p90": stats["p90"],
                "psi": psi, "drift": drift, "r": run_id,
            },
        )
        conn.commit()
    logger.info("Wrote audit trail to PostgreSQL for run %s (PSI=%s)", run_id, psi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pandas batch scorer (deployed/CI path)")
    parser.add_argument("--input", default=None, help="Input Parquet (default: configured)")
    parser.add_argument("--skip-postgres", action="store_true", help="Write Parquet only")
    parser.add_argument(
        "--run-id",
        default=f"run-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}",
    )
    args = parser.parse_args()

    try:
        result = score(args.run_id, input_path=args.input, skip_postgres=args.skip_postgres)
        logger.info("Done: %s", result)
    except Exception:
        logger.exception("Batch scoring FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
