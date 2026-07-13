"""
Score Distribution Monitor
============================
Tracks how the churn score distribution changes between consecutive batch runs.

Metric used: Population Stability Index (PSI)
  PSI is the standard metric for model output monitoring in banking/finance
  (Basel II regulation). It measures how much a distribution has shifted:

  PSI = Σ (actual_% - reference_%) × ln(actual_% / reference_%)

  Thresholds:
    PSI < 0.10  — No significant change
    0.10–0.20   — Moderate change, monitor
    PSI > 0.20  — Significant shift, investigate (may indicate data drift,
                  upstream feature change, or model degradation)

Why PSI and not KS test:
  - KS test is for continuous distributions and gives a binary pass/fail
  - PSI gives a continuous magnitude score, making it easier to trend over time
  - PSI is required by Basel II compliance — relevant for finance/banking portfolios
  - PSI is directional: you can see if scores shifted upward or downward

Why monitor batch scores specifically:
  - Score distribution drift often predates metric degradation by days/weeks
  - Catching it early allows preemptive retraining before customer harm
  - This is how LinkedIn and Airbnb monitor their nightly scoring pipelines

Usage:
    from monitoring.score_monitor import ScoreMonitor
    monitor = ScoreMonitor()
    psi, flagged = monitor.compute_and_store_psi(run_id, scored_parquet_path)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logger = logging.getLogger(__name__)


class ScoreMonitor:
    """
    Monitors churn score distribution stability across batch runs.

    PSI is computed by binning scores into equal-width buckets
    (default: 10 bins from 0.0 to 1.0) and measuring the relative
    shift between runs.
    """

    def __init__(self):
        self.cfg = settings.pipeline
        self.n_bins = self.cfg.psi_n_bins
        self.threshold = self.cfg.psi_threshold
        self.bin_edges = np.linspace(0.0, 1.0, self.n_bins + 1)

    # -------------------------------------------------------------------------
    # PSI computation
    # -------------------------------------------------------------------------

    def _bin_distribution(self, scores: np.ndarray) -> np.ndarray:
        """
        Compute the fraction of scores in each bin.
        Returns array of shape (n_bins,) that sums to 1.0.
        """
        counts, _ = np.histogram(scores, bins=self.bin_edges)
        # Add epsilon to avoid log(0) — same convention as Basel II
        fractions = (counts + 1e-6) / (len(scores) + self.n_bins * 1e-6)
        return fractions

    def compute_psi(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
    ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute Population Stability Index between two score arrays.

        Parameters
        ----------
        reference_scores : Scores from the previous (reference) batch run
        current_scores   : Scores from the current batch run

        Returns
        -------
        psi : float — overall PSI value
        ref_dist : np.ndarray — reference bin fractions (for plotting)
        cur_dist : np.ndarray — current bin fractions (for plotting)
        """
        ref_dist = self._bin_distribution(reference_scores)
        cur_dist = self._bin_distribution(current_scores)

        # PSI formula: Σ (cur - ref) × ln(cur / ref)
        psi_per_bin = (cur_dist - ref_dist) * np.log(cur_dist / ref_dist)
        psi = float(np.sum(psi_per_bin))

        return psi, ref_dist, cur_dist

    def psi_interpretation(self, psi: float) -> str:
        if psi < 0.10:
            return "stable"
        elif psi < 0.20:
            return "moderate_change"
        else:
            return "significant_shift"

    # -------------------------------------------------------------------------
    # Main interface
    # -------------------------------------------------------------------------

    def compute_and_store_psi(
        self,
        run_id: str,
        current_output_path: str,
    ) -> Tuple[Optional[float], bool]:
        """
        Compute PSI vs the previous batch run and store the result.

        1. Load current run's scores from Parquet
        2. Load previous run's scores from PostgreSQL (most recent before this run)
        3. Compute PSI
        4. Optionally persist comparison stats

        Returns
        -------
        psi : float or None (None if no previous run exists)
        drift_flagged : bool (True if PSI > threshold)
        """
        # Load current scores
        current_df = pd.read_parquet(
            current_output_path,
            columns=["churn_probability"],
        )
        current_scores = current_df["churn_probability"].values
        logger.info(
            f"Current scores loaded: {len(current_scores):,} rows, "
            f"mean={current_scores.mean():.4f}"
        )

        # Load previous run's scores from PostgreSQL
        previous_scores = self._load_previous_scores(run_id)

        if previous_scores is None:
            logger.info(
                "No previous batch run found — PSI baseline not established. "
                "PSI will be computed from the next run onwards."
            )
            return None, False

        logger.info(
            f"Reference scores loaded: {len(previous_scores):,} rows, "
            f"mean={previous_scores.mean():.4f}"
        )

        # Compute PSI
        psi, ref_dist, cur_dist = self.compute_psi(previous_scores, current_scores)
        interpretation = self.psi_interpretation(psi)
        drift_flagged = psi > self.threshold

        logger.info(
            f"PSI computed: {psi:.4f} | "
            f"Interpretation: {interpretation} | "
            f"Drift flagged: {drift_flagged}"
        )

        if drift_flagged:
            logger.warning(
                f"DRIFT ALERT: PSI={psi:.4f} exceeds threshold={self.threshold}. "
                "Score distribution has shifted significantly. "
                "Investigate data pipeline and consider retraining."
            )

        # Log per-bin breakdown
        logger.info("Per-bin PSI breakdown:")
        for i, (ref, cur) in enumerate(zip(ref_dist, cur_dist)):
            lo = i / self.n_bins
            hi = (i + 1) / self.n_bins
            bin_psi = (cur - ref) * np.log(cur / ref) if ref > 0 else 0
            direction = "+" if cur > ref else "-"
            logger.info(
                f"  [{lo:.1f}-{hi:.1f}] ref={ref:.4f} cur={cur:.4f} "
                f"psi={bin_psi:.4f} {direction}"
            )

        return psi, drift_flagged

    # -------------------------------------------------------------------------
    # PostgreSQL integration
    # -------------------------------------------------------------------------

    def _load_previous_scores(
        self,
        current_run_id: str,
        n_sample: int = 200_000,
    ) -> Optional[np.ndarray]:
        """
        Load a sample of scores from the run immediately before current_run_id.
        Uses a sample (not all 1M rows) for PSI efficiency — statistically
        sufficient for distribution comparison.
        """
        try:
            from sqlalchemy import text
            from db.connection import _sync_engine

            with _sync_engine.connect() as conn:
                # Find the previous completed run_id
                prev_result = conn.execute(
                    text("""
                    SELECT run_id
                    FROM batch_runs
                    WHERE status IN ('completed', 'validated')
                      AND run_id != :current_run_id
                    ORDER BY started_at DESC
                    LIMIT 1
                """),
                    {"current_run_id": current_run_id},
                )

                prev_row = prev_result.fetchone()
                if not prev_row:
                    return None

                prev_run_id = prev_row[0]
                logger.info(f"Using previous run as reference: {prev_run_id}")

                # Sample scores for PSI computation
                scores_result = conn.execute(
                    text("""
                    SELECT churn_probability
                    FROM predictions
                    WHERE run_id = :prev_run_id
                    TABLESAMPLE SYSTEM(20)   -- sample ~20% for PSI efficiency
                    LIMIT :n_sample
                """),
                    {"prev_run_id": prev_run_id, "n_sample": n_sample},
                )

                rows = scores_result.fetchall()
                if not rows:
                    return None

                return np.array([float(row[0]) for row in rows])

        except Exception as e:
            logger.warning(f"Could not load previous scores from PostgreSQL: {e}")
            return None

    # -------------------------------------------------------------------------
    # Reporting helpers (used by Gradio dashboard)
    # -------------------------------------------------------------------------

    def get_psi_history(self, n_runs: int = 10) -> pd.DataFrame:
        """
        Retrieve PSI history for the last N completed runs.
        Used by the Gradio monitoring dashboard.
        """
        try:
            from sqlalchemy import text
            from db.connection import _sync_engine

            with _sync_engine.connect() as conn:
                result = conn.execute(
                    text("""
                    SELECT
                        run_id,
                        started_at,
                        psi_vs_previous,
                        drift_flagged,
                        score_mean,
                        score_p50,
                        records_scored
                    FROM batch_runs
                    WHERE status IN ('completed', 'validated')
                      AND psi_vs_previous IS NOT NULL
                    ORDER BY started_at DESC
                    LIMIT :n
                """),
                    {"n": n_runs},
                )

                rows = result.fetchall()
                if not rows:
                    return pd.DataFrame()

                return pd.DataFrame(
                    rows,
                    columns=[
                        "run_id",
                        "started_at",
                        "psi_vs_previous",
                        "drift_flagged",
                        "score_mean",
                        "score_p50",
                        "records_scored",
                    ],
                )

        except Exception as e:
            logger.warning(f"Could not load PSI history: {e}")
            return pd.DataFrame()

    def get_score_distribution_comparison(
        self,
        run_id_a: str,
        run_id_b: str,
    ) -> pd.DataFrame:
        """
        Return side-by-side score histograms for two runs.
        Used by Gradio to render the comparison chart.
        """
        try:
            from sqlalchemy import text
            from db.connection import _sync_engine

            with _sync_engine.connect() as conn:
                result = conn.execute(
                    text("""
                    SELECT run_id, bin, bin_lower, bin_upper, count, fraction
                    FROM v_score_histogram
                    WHERE run_id IN (:id_a, :id_b)
                    ORDER BY run_id, bin
                """),
                    {"id_a": run_id_a, "id_b": run_id_b},
                )

                rows = result.fetchall()
                return pd.DataFrame(
                    rows,
                    columns=[
                        "run_id",
                        "bin",
                        "bin_lower",
                        "bin_upper",
                        "count",
                        "fraction",
                    ],
                )

        except Exception as e:
            logger.warning(f"Could not load score distributions: {e}")
            return pd.DataFrame()
