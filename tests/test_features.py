"""Tests for the shared feature layer — the thing that prevents train/serve skew."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from features import (  # noqa: E402
    CATEGORICAL_COLS,
    DROP_COLS,
    TARGET_COL,
    build_feature_matrix,
    encode_categoricals,
    engineer_features,
    predict_scores,
)


def _raw():
    return pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "bd": [30, 999],
            "registration_days": [1000, 200],
            "n_transactions": [10, 1],
            "total_paid": [1490, 149],
            "avg_plan_price": [149, 149],
            "avg_plan_days": [30, 30],
            "total_discount": [149, 0],
            "n_auto_renew": [10, 0],
            "n_cancels": [0, 1],
            "membership_tenure_days": [300, 30],
            "days_to_expire": [15, -5],
            "city": [1, 13],
            "gender": ["male", None],
            "registered_via": [7, 9],
            "payment_method_id": [41, 40],
            "last_is_auto_renew": [1, 0],
            "last_is_cancel": [0, 1],
            "is_churn": [0, 1],
        }
    )


def test_engineered_flags_and_dirty_age():
    eng = engineer_features(_raw())
    assert np.isnan(eng.loc[1, "bd"]), "dirty bd=999 must be nulled"
    assert eng.loc[1, "is_expired"] == 1  # days_to_expire = -5
    assert eng.loc[1, "has_cancelled"] == 1
    assert eng.loc[0, "cancel_rate"] == 0.0


def test_train_infer_parity_no_skew():
    """The core guarantee: training and inference produce identical matrices."""
    raw = _raw()
    train_df, encoders = encode_categoricals(engineer_features(raw), CATEGORICAL_COLS, fit=True)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    X = build_feature_matrix(raw.drop(columns=[TARGET_COL]), encoders, feature_cols)
    assert list(X.columns) == feature_cols
    assert np.isnan(X.loc[1, "bd"]), "NaN must reach LightGBM, not be imputed"
    shared = [c for c in feature_cols if c != "bd"]
    assert np.allclose(train_df[shared].to_numpy("float32"), X[shared].to_numpy("float32"))


def test_unseen_category_maps_safely():
    raw = _raw()
    _, encoders = encode_categoricals(engineer_features(raw), CATEGORICAL_COLS, fit=True)
    novel = raw.copy()
    novel.loc[0, "payment_method_id"] = 9999  # never seen at fit time
    out, _ = encode_categoricals(engineer_features(novel), CATEGORICAL_COLS, encoders=encoders, fit=False)
    assert out["payment_method_id"].notna().all()  # did not crash / produce NaN


class _StubModel:
    """predict_proba returning controlled probabilities to test output derivation."""

    def __init__(self, probs):
        self._probs = np.asarray(probs)

    def predict_proba(self, X):
        p = self._probs[: len(X)]
        return np.column_stack([1 - p, p])


def test_predict_scores_derivation():
    raw = _raw()
    train_df, encoders = encode_categoricals(engineer_features(raw), CATEGORICAL_COLS, fit=True)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

    model = _StubModel([0.05, 0.85])
    out = predict_scores(model, raw.drop(columns=[TARGET_COL]), encoders, feature_cols)

    assert list(out["customer_id"]) == ["a", "b"]
    assert list(out["churn_label"]) == [False, True]         # 0.5 threshold
    assert list(out["risk_tier"]) == ["low", "high"]         # 0.40 / 0.70 cuts
    assert list(out["churn_decile"]) == [1, 9]               # ceil(p*10), clipped 1..10
