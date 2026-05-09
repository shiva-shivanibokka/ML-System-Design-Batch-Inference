"""
Synthetic Customer Dataset Generator
=====================================
Generates 1,000,000 realistic customer records for churn prediction.

Features engineered to reflect real SaaS/telecom churn dynamics:
  - Churn probability increases with: high monthly charges, payment failures,
    low tenure, many support calls, month-to-month contracts
  - Churn probability decreases with: long tenure, multiple products,
    tech support, annual contracts

Usage:
    python data/generate_data.py                     # generates 1M rows
    python data/generate_data.py --n 100000          # generate 100K (for testing)
    python data/generate_data.py --seed 123          # custom random seed

Output:
    data/customers.parquet   (Parquet, ~120MB for 1M rows)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Allow running as a script from project root
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Categorical value pools (kept consistent with config.yaml)
# ---------------------------------------------------------------------------
CONTRACT_TYPES = ["month-to-month", "one-year", "two-year"]
CONTRACT_WEIGHTS = [0.55, 0.25, 0.20]  # month-to-month is most common

PAYMENT_METHODS = ["credit_card", "bank_transfer", "electronic_check", "mailed_check"]
PAYMENT_WEIGHTS = [0.30, 0.30, 0.25, 0.15]

INTERNET_SERVICES = ["fiber", "dsl", "none"]
INTERNET_WEIGHTS = [0.45, 0.40, 0.15]

REGIONS = ["north", "south", "east", "west", "central"]
REGION_WEIGHTS = [0.20, 0.22, 0.21, 0.19, 0.18]


# ---------------------------------------------------------------------------
# Core generator
# ---------------------------------------------------------------------------


def _compute_churn_probability(df: pd.DataFrame) -> np.ndarray:
    """
    Compute a realistic churn probability for each row using a log-odds model.
    Coefficients are designed to produce ~18.5% base churn rate.

    This is the "ground truth" used to generate the actual_churn label.
    The LightGBM model will then learn to approximate this signal from features.
    """
    # Start from a base log-odds
    log_odds = np.full(len(df), -1.5)  # base ~18% churn

    # --- Strong churn drivers ---
    # Month-to-month contracts: biggest churn predictor
    log_odds += np.where(df["contract_type"] == "month-to-month", 1.20, 0.0)
    log_odds += np.where(df["contract_type"] == "one-year", 0.20, 0.0)
    # (two-year is reference: 0.0)

    # Electronic check payment: associated with higher churn
    log_odds += np.where(df["payment_method"] == "electronic_check", 0.45, 0.0)

    # Payment failures: very strong churn signal
    log_odds += np.clip(df["payment_failures"].values * 0.55, 0, 2.5)

    # Support calls: more calls = more frustrated = more churn
    log_odds += np.clip(df["num_support_calls"].values * 0.18, 0, 1.2)

    # High monthly charges relative to tenure
    charges_per_month = df["monthly_charges"].values
    log_odds += np.where(charges_per_month > 80, 0.40, 0.0)
    log_odds += np.where(charges_per_month > 100, 0.30, 0.0)

    # Long time since last login (disengaged)
    log_odds += np.clip(df["days_since_last_login"].values / 90.0, 0, 1.0) * 0.60

    # --- Churn reducers ---
    # Long tenure: loyal customers don't churn
    log_odds -= np.clip(df["tenure_months"].values / 24.0, 0, 1.0) * 1.10

    # Multiple products: switching cost
    log_odds -= np.clip((df["num_products"].values - 1) * 0.22, 0, 1.0)

    # Tech support: reduces frustration
    log_odds -= np.where(df["has_tech_support"].values, 0.35, 0.0)

    # Referrals given: brand advocates don't churn
    log_odds -= np.clip(df["referrals_given"].values * 0.25, 0, 0.8)

    # High session time: engaged users stay
    avg_session = df["avg_session_minutes"].values
    log_odds -= np.clip(avg_session / 60.0, 0, 1.0) * 0.45

    # No internet: lower churn (fewer reasons to leave)
    log_odds -= np.where(df["internet_service"] == "none", 0.50, 0.0)

    # Convert log-odds → probability
    return 1.0 / (1.0 + np.exp(-log_odds))


def generate_customers(
    n: int,
    seed: int = 42,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """
    Generate n synthetic customer records in chunks to keep memory manageable.

    Parameters
    ----------
    n         : Total number of customers to generate.
    seed      : Random seed for reproducibility.
    chunksize : Rows generated per chunk (controls peak RAM usage).

    Returns
    -------
    pd.DataFrame with columns matching config.yaml feature definitions.
    """
    rng = np.random.default_rng(seed)
    fake = Faker()
    Faker.seed(seed)

    chunks = []
    n_chunks = (n + chunksize - 1) // chunksize

    logger.info(f"Generating {n:,} customers in {n_chunks} chunks of {chunksize:,}...")

    with tqdm(total=n, unit="rows", unit_scale=True) as pbar:
        generated = 0
        chunk_idx = 0

        while generated < n:
            size = min(chunksize, n - generated)

            # --- Demographics ---
            age = rng.integers(18, 91, size=size).astype("int16")

            # --- Account ---
            tenure_months = rng.integers(0, 73, size=size).astype("int16")
            contract_type = rng.choice(CONTRACT_TYPES, size=size, p=CONTRACT_WEIGHTS)
            payment_method = rng.choice(PAYMENT_METHODS, size=size, p=PAYMENT_WEIGHTS)

            # --- Services ---
            internet_service = rng.choice(
                INTERNET_SERVICES, size=size, p=INTERNET_WEIGHTS
            )
            has_phone_service = rng.choice([True, False], size=size, p=[0.90, 0.10])
            has_streaming = rng.choice([True, False], size=size, p=[0.60, 0.40])
            has_tech_support = rng.choice([True, False], size=size, p=[0.45, 0.55])

            # --- Billing ---
            # Monthly charges: fiber > DSL > none; distributions are realistic
            monthly_charges = (
                np.where(
                    internet_service == "fiber",
                    rng.normal(85, 20, size=size),
                    np.where(
                        internet_service == "dsl",
                        rng.normal(55, 15, size=size),
                        rng.normal(25, 8, size=size),  # no internet
                    ),
                )
                .clip(15, 150)
                .round(2)
            )

            total_charges = (monthly_charges * (tenure_months + 1)).round(2)

            payment_failures = (
                np.where(
                    payment_method == "electronic_check",
                    rng.poisson(1.5, size=size),
                    rng.poisson(0.3, size=size),
                )
                .clip(0, 10)
                .astype("int16")
            )

            # --- Engagement ---
            num_products = rng.integers(1, 8, size=size).astype("int16")

            # Support calls: correlated with payment failures + contract type
            base_calls = np.where(contract_type == "month-to-month", 1.5, 0.6)
            num_support_calls = (
                (rng.poisson(base_calls, size=size) + payment_failures * 0.4)
                .clip(0, 20)
                .round(0)
                .astype("int16")
            )

            avg_session_minutes = rng.exponential(35, size=size).clip(0, 240).round(2)

            # Days since last login: higher for disengaged customers
            days_since_last_login = (
                np.where(
                    contract_type == "month-to-month",
                    rng.exponential(25, size=size),
                    rng.exponential(8, size=size),
                )
                .clip(0, 365)
                .round(0)
                .astype("int16")
            )

            referrals_given = rng.poisson(0.8, size=size).clip(0, 10).astype("int16")

            region = rng.choice(REGIONS, size=size, p=REGION_WEIGHTS)

            # --- Unique customer IDs ---
            customer_ids = [f"CUST-{chunk_idx:04d}-{i:06d}" for i in range(size)]

            # --- Build chunk DataFrame ---
            chunk_df = pd.DataFrame(
                {
                    "customer_id": customer_ids,
                    "age": age,
                    "region": region,
                    "tenure_months": tenure_months,
                    "contract_type": contract_type,
                    "payment_method": payment_method,
                    "internet_service": internet_service,
                    "has_phone_service": has_phone_service,
                    "has_streaming": has_streaming,
                    "has_tech_support": has_tech_support,
                    "monthly_charges": monthly_charges,
                    "total_charges": total_charges,
                    "payment_failures": payment_failures,
                    "num_products": num_products,
                    "num_support_calls": num_support_calls,
                    "avg_session_minutes": avg_session_minutes,
                    "days_since_last_login": days_since_last_login,
                    "referrals_given": referrals_given,
                }
            )

            # --- Generate churn labels ---
            churn_prob = _compute_churn_probability(chunk_df)
            # Add noise so it's not perfectly deterministic
            noisy_prob = np.clip(churn_prob + rng.normal(0, 0.05, size=size), 0, 1)
            chunk_df["actual_churn"] = rng.uniform(0, 1, size=size) < noisy_prob

            chunks.append(chunk_df)
            generated += size
            chunk_idx += 1
            pbar.update(size)

    df = pd.concat(chunks, ignore_index=True)

    # Report churn rate
    churn_rate = df["actual_churn"].mean()
    logger.info(f"Generated {len(df):,} customers | Churn rate: {churn_rate:.1%}")

    return df


def save_dataset(df: pd.DataFrame, output_path: str) -> None:
    """Save to Parquet (columnar format — optimal for Spark reads)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Cast types for Parquet efficiency
    df["has_phone_service"] = df["has_phone_service"].astype(bool)
    df["has_streaming"] = df["has_streaming"].astype(bool)
    df["has_tech_support"] = df["has_tech_support"].astype(bool)
    df["actual_churn"] = df["actual_churn"].astype(bool)

    df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    logger.info(f"Saved to {output_path} ({size_mb:.1f} MB, {len(df):,} rows)")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic customer dataset")
    parser.add_argument(
        "--n",
        type=int,
        default=settings.data.n_customers,
        help=f"Number of customers to generate (default: {settings.data.n_customers:,})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=settings.data.random_seed,
        help=f"Random seed (default: {settings.data.random_seed})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=settings.data.output_path,
        help=f"Output Parquet path (default: {settings.data.output_path})",
    )
    args = parser.parse_args()

    t0 = time.perf_counter()
    df = generate_customers(n=args.n, seed=args.seed)
    save_dataset(df, args.output)
    elapsed = time.perf_counter() - t0

    logger.info(f"Done in {elapsed:.1f}s  ({args.n / elapsed:,.0f} rows/sec)")

    # Print summary statistics
    print("\n--- Dataset Summary ---")
    print(f"  Shape         : {df.shape}")
    print(f"  Churn rate    : {df['actual_churn'].mean():.1%}")
    print(
        f"  Contract dist : {df['contract_type'].value_counts(normalize=True).round(3).to_dict()}"
    )
    print(
        f"  Region dist   : {df['region'].value_counts(normalize=True).round(3).to_dict()}"
    )
    print(
        f"  Monthly charges: mean={df['monthly_charges'].mean():.2f}, std={df['monthly_charges'].std():.2f}"
    )
    print(
        f"  Tenure months  : mean={df['tenure_months'].mean():.1f}, max={df['tenure_months'].max()}"
    )
    print()


if __name__ == "__main__":
    main()
