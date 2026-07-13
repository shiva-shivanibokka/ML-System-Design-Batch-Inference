"""
Shared feature engineering — the single source of truth.
========================================================
Training (models/train.py), Spark inference (spark/batch_inference.py), and the
benchmark (bench/compare.py) ALL import from here. Previously each had its
own hand-copied `engineer_features`, which is a classic train/serve-skew bug:
if two copies drift, you get silently wrong predictions and no error anywhere.
One function, imported everywhere, makes that class of bug impossible.

Input is a customer-level DataFrame as produced by data/build_dataset.py
(one row per member, KKBox schema). Output is a model-ready matrix.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Columns that are never model features (identifier + target).
ID_COL = "customer_id"
TARGET_COL = "is_churn"
DROP_COLS = [ID_COL, TARGET_COL]

# Categorical columns (KKBox codes / flags). Kept in sync with config.yaml.
CATEGORICAL_COLS = [
    "city",
    "gender",
    "registered_via",
    "payment_method_id",
    "last_is_auto_renew",
    "last_is_cancel",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived ratio/flag features on top of the raw KKBox columns.
    Must be deterministic and column-order independent — the Spark job applies
    this per partition and the result must match training exactly.
    """
    df = df.copy()

    # KKBox `bd` (age) is notoriously dirty: negatives and values >100. Clip to a
    # plausible range and treat out-of-range as unknown (median-imputed downstream).
    if "bd" in df.columns:
        df["bd"] = df["bd"].where((df["bd"] >= 0) & (df["bd"] <= 100), np.nan)

    # Ratio features — guard every denominator with +1 (no div-by-zero).
    df["discount_rate"] = (df["total_discount"] / (df["total_paid"].abs() + 1)).round(4)
    df["paid_per_txn"] = (df["total_paid"] / (df["n_transactions"] + 1)).round(4)
    df["cancel_rate"] = (df["n_cancels"] / (df["n_transactions"] + 1)).round(4)
    df["autorenew_rate"] = (df["n_auto_renew"] / (df["n_transactions"] + 1)).round(4)

    # Binary risk flags — cheap signal LightGBM can split on directly.
    df["is_expired"] = (df["days_to_expire"] < 0).astype("int8")
    df["never_autorenew"] = (df["n_auto_renew"] == 0).astype("int8")
    df["has_cancelled"] = (df["n_cancels"] > 0).astype("int8")

    return df


def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    encoders: Optional[Dict[str, LabelEncoder]] = None,
    fit: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Label-encode categorical columns.

    fit=True  → fit new encoders (training).
    fit=False → reuse provided encoders (inference). Unseen categories map to the
                first known class so a novel value never crashes a nightly run.
    """
    df = df.copy()
    if encoders is None:
        encoders = {}

    for col in categorical_cols:
        if col not in df.columns:
            continue
        # Normalise: everything to string, NaN to a literal "unknown" bucket.
        col_values = df[col].astype(str).replace({"nan": "unknown", "None": "unknown"})
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(col_values)
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            safe = col_values.apply(lambda x: x if x in known else le.classes_[0])
            df[col] = le.transform(safe)

    return df, encoders


def build_feature_matrix(
    df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    End-to-end transform used at INFERENCE time (Spark partitions, benchmark).
    engineer → encode → align to the exact training column order → float32.

    NaN (e.g. a clipped-out `bd`) is deliberately left in place: LightGBM routes
    NaN natively and identically at train and inference time. Imputing here with a
    per-partition median would instead make a customer's score depend on which
    Spark partition they landed in — a real train/serve skew. So: no imputation.
    """
    df = engineer_features(df)
    df, _ = encode_categoricals(df, CATEGORICAL_COLS, encoders=encoders, fit=False)
    X = df.reindex(columns=feature_cols)
    return X.apply(pd.to_numeric, errors="coerce").astype("float32")


def predict_scores(model, df: pd.DataFrame, encoders, feature_cols: List[str]) -> pd.DataFrame:
    """
    Score a customer-level frame → the canonical output columns.

    Shared by the Spark job (per partition) and the pandas scorer (score_batch.py)
    so the probability→label/decile/risk_tier derivation lives in exactly one place.
    """
    ids = df[ID_COL].to_numpy()
    X = build_feature_matrix(df, encoders, feature_cols)
    prob = model.predict_proba(X)[:, 1]
    return pd.DataFrame(
        {
            ID_COL: ids,
            "churn_probability": np.round(prob, 4).astype(float),
            "churn_label": prob >= 0.5,
            "churn_decile": np.clip(np.ceil(prob * 10).astype(int), 1, 10),
            "risk_tier": np.where(
                prob >= 0.70, "high", np.where(prob >= 0.40, "medium", "low")
            ),
        }
    )


def demo() -> None:
    """Runnable self-check: engineered features are correct and train/infer agree."""
    raw = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "bd": [30, 999],  # 999 is dirty → should become NaN then imputed
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

    eng = engineer_features(raw)
    assert np.isnan(eng.loc[1, "bd"]), "dirty bd=999 must be nulled before impute"
    assert eng.loc[0, "cancel_rate"] == 0.0
    assert eng.loc[1, "has_cancelled"] == 1
    assert eng.loc[1, "is_expired"] == 1  # days_to_expire=-5

    # Fit on training, then transform the same rows at inference → identical matrix.
    train_df, encoders = encode_categoricals(engineer_features(raw), CATEGORICAL_COLS, fit=True)
    feature_cols = [c for c in train_df.columns if c not in DROP_COLS]
    X_infer = build_feature_matrix(raw.drop(columns=[TARGET_COL]), encoders, feature_cols)
    assert list(X_infer.columns) == feature_cols, "inference column order must match training"
    assert X_infer.shape == (2, len(feature_cols))
    assert np.isnan(X_infer.loc[1, "bd"]), "NaN must pass through to LightGBM, not be imputed"
    # Encoded features and column order must be identical between the two paths.
    shared = [c for c in feature_cols if c not in ("bd",)]
    assert np.allclose(
        train_df[shared].to_numpy("float32"), X_infer[shared].to_numpy("float32")
    ), "train and inference matrices must agree (no skew)"
    print(f"features.demo OK — {len(feature_cols)} model features, train==infer, NaN preserved")


if __name__ == "__main__":
    demo()
