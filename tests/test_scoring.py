"""Tests for the deployed pandas scorer's validation gates."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from score_batch import _validate  # noqa: E402


def _scored(probs):
    probs = np.asarray(probs, dtype=float)
    return pd.DataFrame({"churn_probability": probs, "churn_label": probs >= 0.5})


def test_healthy_batch_passes():
    rng = np.random.default_rng(0)
    _validate(_scored(rng.uniform(0, 1, 1000)), n_read=1000)  # no raise


def test_degenerate_distribution_rejected():
    with pytest.raises(ValueError, match="degenerate"):
        _validate(_scored([0.3] * 1000), n_read=1000)


def test_dropped_rows_rejected():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="dropped"):
        _validate(_scored(rng.uniform(0, 1, 500)), n_read=1000)  # only half scored
