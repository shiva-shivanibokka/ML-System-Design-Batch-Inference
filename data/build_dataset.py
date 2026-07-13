"""
Build the customer-level dataset from raw KKBox CSVs.
=====================================================
Joins three real files into one row per member:

    train_v2.csv         → the churn label (is_churn) and the member universe
    members_v3.csv       → demographics (city, bd/age, gender, registered_via, ...)
    transactions_v2.csv  → subscription/payment behaviour, aggregated per member

The ~30GB user_logs are intentionally NOT used — members + transactions already
carry the strongest churn signal (auto-renew, cancels, plan price, expiry), and
this keeps the whole thing runnable on one machine.

No leakage: transactions are filtered to on/before the reference date, and the
label describes churn AFTER it.

Output: data/customers.parquet  (~970K rows, the table the pipeline scores).

Usage:
    python data/build_dataset.py
    python data/build_dataset.py --limit 50000     # small slice for CI / testing
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _to_datetime(series: pd.Series) -> pd.Series:
    """KKBox dates are ints like 20170228. Parse to datetime (invalid → NaT)."""
    return pd.to_datetime(series, format="%Y%m%d", errors="coerce")


def aggregate_transactions(tx: pd.DataFrame, reference: pd.Timestamp) -> pd.DataFrame:
    """
    Collapse the transaction history to one row per member.
    Only transactions on/before the reference date are used (no leakage).
    """
    tx = tx.copy()
    tx["transaction_date"] = _to_datetime(tx["transaction_date"])
    tx["membership_expire_date"] = _to_datetime(tx["membership_expire_date"])
    tx = tx[tx["transaction_date"] <= reference]

    tx = tx.sort_values(["msno", "transaction_date"])
    grouped = tx.groupby("msno", sort=False)

    # Behaviour aggregates.
    agg = grouped.agg(
        n_transactions=("msno", "size"),
        total_paid=("actual_amount_paid", "sum"),
        avg_plan_price=("plan_list_price", "mean"),
        avg_plan_days=("payment_plan_days", "mean"),
        n_auto_renew=("is_auto_renew", "sum"),
        n_cancels=("is_cancel", "sum"),
        first_txn=("transaction_date", "min"),
        last_expire=("membership_expire_date", "max"),
    )

    # Latest-transaction snapshot (last row per member after the sort above).
    last = grouped.tail(1).set_index("msno")
    agg["payment_method_id"] = last["payment_method_id"]
    agg["last_is_auto_renew"] = last["is_auto_renew"]
    agg["last_is_cancel"] = last["is_cancel"]

    # total_discount = list price minus what was actually paid, summed per member.
    discount = tx["plan_list_price"] - tx["actual_amount_paid"]
    agg["total_discount"] = discount.groupby(tx["msno"]).sum()

    # Derived durations, in days, relative to the reference cutoff.
    agg["membership_tenure_days"] = (agg["last_expire"] - agg["first_txn"]).dt.days
    agg["days_to_expire"] = (agg["last_expire"] - reference).dt.days

    return agg.drop(columns=["first_txn", "last_expire"]).reset_index()


def build_customer_dataset(limit: int | None = None) -> pd.DataFrame:
    cfg = settings.data
    raw = Path(cfg.raw_dir)
    reference = pd.to_datetime(str(cfg.reference_date), format="%Y%m%d")

    for f in (cfg.labels_file, cfg.members_file, cfg.transactions_file):
        if not (raw / f).exists():
            raise FileNotFoundError(
                f"{raw / f} not found. Run `python data/download_kkbox.py` first."
            )

    logger.info("Loading labels (%s)...", cfg.labels_file)
    labels = pd.read_csv(raw / cfg.labels_file)
    if limit:
        labels = labels.head(limit)
    logger.info("  %s labelled members | churn rate %.1f%%",
                f"{len(labels):,}", 100 * labels["is_churn"].mean())

    logger.info("Loading members (%s)...", cfg.members_file)
    members = pd.read_csv(raw / cfg.members_file)
    members["registration_days"] = (
        reference - _to_datetime(members["registration_init_time"])
    ).dt.days
    members = members[["msno", "city", "bd", "gender", "registered_via", "registration_days"]]

    logger.info("Loading + aggregating transactions (%s)...", cfg.transactions_file)
    tx = pd.read_csv(raw / cfg.transactions_file)
    tx_agg = aggregate_transactions(tx, reference)

    logger.info("Joining...")
    df = labels.merge(members, on="msno", how="left").merge(tx_agg, on="msno", how="left")

    # Members with no qualifying transaction get 0-filled behaviour (real signal:
    # a member we have demographics for but no payment on/before the cutoff).
    tx_cols = [c for c in tx_agg.columns if c != "msno"]
    df[tx_cols] = df[tx_cols].fillna(0)

    df = df.rename(columns={"msno": "customer_id"})

    # Final column order: id, features, label.
    feature_cols = cfg.numeric_features + cfg.categorical_features
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Built dataset is missing expected feature columns: {missing}")
    df = df[["customer_id"] + feature_cols + ["is_churn"]]

    # Compact dtypes for a smaller Parquet.
    for c in cfg.categorical_features + ["is_churn"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int32") if c != "gender" else df[c]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build customer-level dataset from KKBox CSVs")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N labelled members (CI / testing)")
    parser.add_argument("--output", type=str, default=settings.data.output_path)
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = build_customer_dataset(limit=args.limit)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False, engine="pyarrow", compression="snappy")

    size_mb = out.stat().st_size / (1024 * 1024)
    logger.info("Saved %s (%.1f MB, %s rows) in %.1fs",
                out, size_mb, f"{len(df):,}", time.perf_counter() - t0)
    print("\n--- Dataset Summary ---")
    print(f"  Rows        : {len(df):,}")
    print(f"  Churn rate  : {df['is_churn'].mean():.1%}")
    print(f"  Columns     : {list(df.columns)}")


if __name__ == "__main__":
    main()
