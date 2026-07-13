"""Tests for the KKBox ETL — aggregation correctness and the no-leakage cutoff."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.build_dataset import aggregate_transactions  # noqa: E402


def test_aggregate_transactions_and_leakage():
    reference = pd.Timestamp("2017-03-01")
    tx = pd.DataFrame(
        {
            "msno": ["u1", "u1", "u1", "u2"],
            "payment_method_id": [40, 41, 41, 36],
            "plan_list_price": [149, 149, 149, 99],
            "payment_plan_days": [30, 30, 30, 30],
            "actual_amount_paid": [149, 99, 149, 99],  # middle txn had a 50 discount
            "is_auto_renew": [1, 1, 0, 1],
            "is_cancel": [0, 0, 1, 0],
            "transaction_date": [20170101, 20170201, 20170228, 20170401],  # u2's txn is AFTER cutoff
            "membership_expire_date": [20170201, 20170301, 20170310, 20170501],
        }
    )

    agg = aggregate_transactions(tx, reference).set_index("msno")

    # u2's only transaction is after the reference date → dropped entirely (no leakage).
    assert "u2" not in agg.index

    u1 = agg.loc["u1"]
    assert u1["n_transactions"] == 3
    assert u1["total_paid"] == 149 + 99 + 149
    assert u1["total_discount"] == 50            # (149-149)+(149-99)+(149-149)
    assert u1["n_auto_renew"] == 2
    assert u1["n_cancels"] == 1
    assert u1["last_is_cancel"] == 1             # latest transaction (2017-02-28) was a cancel
    assert u1["last_is_auto_renew"] == 0
    # days_to_expire = last expire (2017-03-10) − reference (2017-03-01) = 9
    assert u1["days_to_expire"] == 9
