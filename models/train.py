"""
LightGBM Churn Model — Training Script
========================================
Trains a LightGBM binary classifier on the synthetic customer dataset.

Design decisions:
  - Strict chronological split is not applicable here (cross-sectional data,
    not time series). Uses stratified random split instead.
  - Label encoding for categoricals (LightGBM can handle these natively,
    but we encode explicitly so the Spark inference job can reproduce the
    exact same transformation without LightGBM overhead).
  - Saves: churn_model.pkl, label_encoders.pkl, feature_columns.pkl
    All three artifacts are needed by spark/batch_inference.py.

Usage:
    python models/train.py
    python models/train.py --input data/customers.parquet --eval

Output metrics printed to stdout and saved in models/training_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Path fix — allow running from project root or from models/
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

# Feature engineering lives in ONE place (features.py) so training and the Spark
# inference job can never drift apart. See features.py for the rationale.
from features import (
    CATEGORICAL_COLS,
    DROP_COLS,
    TARGET_COL,
    encode_categoricals,
    engineer_features,
)

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    raise ImportError("lightgbm not installed. Run: pip install lightgbm")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def load_dataset(path: str) -> pd.DataFrame:
    logger.info(f"Loading dataset from {path}...")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


def prepare_data(
    df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    Dict[str, LabelEncoder],
    List[str],
]:
    """
    Full preprocessing pipeline:
    1. Engineer features
    2. Encode categoricals
    3. Stratified split

    Returns X_train, X_val, X_test, y_train, y_val, y_test, encoders, feature_cols
    """
    cfg = settings.model

    df = engineer_features(df)
    df, encoders = encode_categoricals(df, CATEGORICAL_COLS, fit=True)

    feature_cols = [c for c in df.columns if c not in DROP_COLS]
    X = df[feature_cols]
    y = df[TARGET_COL].astype("int8")

    logger.info(f"Feature matrix: {X.shape}, Churn rate: {y.mean():.1%}")

    # Stratified split — preserves class balance in all splits
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(cfg.val_size + cfg.test_size),
        random_state=cfg.lgbm.random_state,
    )
    train_idx, temp_idx = next(sss1.split(X, y))

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=cfg.test_size / (cfg.val_size + cfg.test_size),
        random_state=cfg.lgbm.random_state,
    )
    val_idx, test_idx = next(sss2.split(X.iloc[temp_idx], y.iloc[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    X_train, X_val, X_test = X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx]
    y_train, y_val, y_test = y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx]

    logger.info(
        f"Split sizes — Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, encoders, feature_cols


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> LGBMClassifier:
    """Train LightGBM with early stopping on validation AUC."""
    cfg = settings.model.lgbm

    model = LGBMClassifier(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        num_leaves=cfg.num_leaves,
        max_depth=cfg.max_depth,
        min_child_samples=cfg.min_child_samples,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        class_weight=cfg.class_weight,
        random_state=cfg.random_state,
        n_jobs=cfg.n_jobs,
        verbose=-1,
    )

    logger.info("Training LightGBM model with early stopping...")
    t0 = time.perf_counter()

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        f"Training complete in {elapsed:.1f}s | Best iteration: {model.best_iteration_}"
    )

    return model


def evaluate_model(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    split_name: str = "test",
) -> Dict:
    """Compute classification metrics."""
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype("int8")

    metrics = {
        "split": split_name,
        "n_samples": len(y),
        "auc_roc": round(float(roc_auc_score(y, y_prob)), 4),
        "avg_precision": round(float(average_precision_score(y, y_prob)), 4),
        "accuracy": round(float(accuracy_score(y, y_pred)), 4),
        "f1": round(float(f1_score(y, y_pred)), 4),
        "precision": round(float(precision_score(y, y_pred)), 4),
        "recall": round(float(recall_score(y, y_pred)), 4),
        "churn_rate_actual": round(float(y.mean()), 4),
        "churn_rate_predicted": round(float(y_pred.mean()), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_artifacts(
    model: LGBMClassifier,
    encoders: Dict[str, LabelEncoder],
    feature_cols: List[str],
) -> None:
    """Save all artifacts needed for inference."""
    cfg = settings.model
    Path(cfg.path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, cfg.path)
    joblib.dump(encoders, cfg.label_encoders_path)
    joblib.dump(feature_cols, cfg.feature_columns_path)

    logger.info(f"Saved model          → {cfg.path}")
    logger.info(f"Saved label encoders → {cfg.label_encoders_path}")
    logger.info(f"Saved feature cols   → {cfg.feature_columns_path}")


def save_training_report(
    train_metrics: Dict,
    val_metrics: Dict,
    test_metrics: Dict,
    model: LGBMClassifier,
    feature_cols: List[str],
    elapsed_secs: float,
) -> None:
    """Save a JSON training report for audit purposes."""
    report = {
        "model_version": settings.model.version,
        "training_time_s": round(elapsed_secs, 2),
        "best_iteration": model.best_iteration_,
        "n_features": len(feature_cols),
        "feature_columns": feature_cols,
        "metrics": {
            "train": train_metrics,
            "val": val_metrics,
            "test": test_metrics,
        },
        # Top 15 feature importances
        "feature_importances": dict(
            sorted(
                zip(feature_cols, model.feature_importances_.tolist()),
                key=lambda x: -x[1],
            )[:15]
        ),
    }

    report_path = Path(settings.model.path).parent / "training_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved training report → {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM churn model")
    parser.add_argument(
        "--input",
        type=str,
        default=settings.data.output_path,
        help=f"Path to customer Parquet (default: {settings.data.output_path})",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Print detailed evaluation after training",
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error(
            f"Dataset not found at {args.input}. "
            "Run `python data/generate_data.py` first."
        )
        sys.exit(1)

    t_total = time.perf_counter()

    # --- Load and prepare ---
    df = load_dataset(args.input)
    X_train, X_val, X_test, y_train, y_val, y_test, encoders, feature_cols = (
        prepare_data(df)
    )

    # --- Train ---
    model = train_model(X_train, y_train, X_val, y_val)

    # --- Evaluate ---
    train_metrics = evaluate_model(model, X_train, y_train, "train")
    val_metrics = evaluate_model(model, X_val, y_val, "val")
    test_metrics = evaluate_model(model, X_test, y_test, "test")

    elapsed = time.perf_counter() - t_total

    # --- Save ---
    save_artifacts(model, encoders, feature_cols)
    save_training_report(
        train_metrics,
        val_metrics,
        test_metrics,
        model,
        feature_cols,
        elapsed,
    )

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Total time        : {elapsed:.1f}s")
    print(f"  Model version     : {settings.model.version}")
    print(f"  Features          : {len(feature_cols)}")
    print(f"  Best iteration    : {model.best_iteration_}")
    print()
    print("  --- Test Set Metrics ---")
    for k, v in test_metrics.items():
        if k not in ("split", "n_samples"):
            print(f"  {k:<26} : {v}")
    print("=" * 60 + "\n")

    if args.eval:
        print("  --- Feature Importances (top 15) ---")
        importances = sorted(
            zip(feature_cols, model.feature_importances_),
            key=lambda x: -x[1],
        )[:15]
        for feat, imp in importances:
            bar = "█" * int(imp / max(i for _, i in importances) * 30)
            print(f"  {feat:<30} {bar} ({imp})")
        print()


if __name__ == "__main__":
    main()
