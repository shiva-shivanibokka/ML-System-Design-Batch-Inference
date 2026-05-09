"""
Batch Inference Monitoring Dashboard
======================================
Gradio 4-tab dashboard for monitoring the nightly batch scoring pipeline.

Tab 1: Batch Run History
  - Table of all pipeline executions with status, duration, record count
  - Score mean/p50 trend chart across runs
  - PSI trend chart (drift over time)
  - Color-coded status badges

Tab 2: Score Distribution
  - Histogram of churn probabilities for any selected run
  - Comparison overlay: current run vs previous run
  - Risk tier breakdown (low / medium / high)
  - Key statistics: mean, std, p10/p50/p90

Tab 3: Benchmark Results
  - Bar chart: PySpark vs pandas vs joblib throughput at each sample size
  - Line chart: duration vs sample size (log-log scale)
  - Speedup table: how much faster PySpark is vs others at 1M records
  - Key insight: when does PySpark's startup cost become worthwhile?

Tab 4: Customer Lookup
  - Enter a customer_id to retrieve their latest churn score
  - Shows probability, decile, risk tier, and which run scored them
  - Bulk lookup: paste up to 20 IDs separated by newlines

All data is fetched from the FastAPI backend via HTTP.
Falls back to demo data if API is not reachable (for demos/screenshots).
"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# API client helpers
# ---------------------------------------------------------------------------

API_BASE = os.getenv("API_BASE_URL", settings.gradio.api_base_url)
TIMEOUT = 10  # seconds


def _get(endpoint: str, params: dict = None) -> Optional[dict]:
    """GET request to FastAPI backend. Returns None on error."""
    try:
        resp = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API call failed ({endpoint}): {e}")
        return None


def _post(endpoint: str, json: dict) -> Optional[dict]:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=json, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.warning(f"API POST failed ({endpoint}): {e}")
        return None


# ---------------------------------------------------------------------------
# Demo / fallback data (used when API is not reachable)
# ---------------------------------------------------------------------------


def _demo_batch_runs() -> pd.DataFrame:
    """Realistic demo data for the Batch Run History tab."""
    import numpy as np

    rng = np.random.default_rng(42)
    n = 8
    now = datetime.now(timezone.utc)

    data = {
        "run_id": [f"run-202401{i + 1:02d}-020000-{i:04x}" for i in range(n)],
        "model_version": ["v1.0.0"] * n,
        "started_at": [
            (now.replace(day=i + 1)).strftime("%Y-%m-%d 02:00:00") for i in range(n)
        ],
        "status": ["validated"] * (n - 1) + ["running"],
        "records_scored": [1_000_000] * (n - 1) + [None],
        "spark_duration_secs": list(rng.uniform(160, 220, n - 1)) + [None],
        "score_mean": list(rng.uniform(0.175, 0.195, n - 1)) + [None],
        "score_p50": list(rng.uniform(0.105, 0.125, n - 1)) + [None],
        "psi_vs_previous": [None] + list(rng.uniform(0.005, 0.15, n - 2)) + [None],
        "drift_flagged": [False] * (n - 1) + [False],
        "validation_passed": [True] * (n - 1) + [None],
        "records_per_second": list(rng.uniform(4800, 6200, n - 1)) + [None],
    }
    return pd.DataFrame(data)


def _demo_score_distribution() -> dict:
    """Demo histogram data."""
    import numpy as np

    rng = np.random.default_rng(42)
    counts = [int(c) for c in rng.integers(30000, 180000, 10)]
    total = sum(counts)
    return {
        "run_id": "run-20240108-020000-0007",
        "n_bins": 10,
        "bins": [
            {
                "bin": i + 1,
                "bin_lower": round(i * 0.1, 1),
                "bin_upper": round((i + 1) * 0.1, 1),
                "count": counts[i],
                "fraction": round(counts[i] / total, 4),
            }
            for i in range(10)
        ],
    }


def _demo_benchmark() -> dict:
    """Demo benchmark results."""
    sizes = [10_000, 100_000, 500_000, 1_000_000]
    results = []
    # PySpark: high startup cost, scales well
    spark_base = [45.0, 52.0, 68.0, 95.0]
    # pandas: fast for small, linear for large
    pandas_base = [0.12, 1.1, 5.5, 11.2]
    # joblib: middle ground
    joblib_base = [0.9, 3.2, 14.8, 30.1]

    for i, size in enumerate(sizes):
        for engine, duration in [
            ("pyspark", spark_base[i]),
            ("pandas", pandas_base[i]),
            ("joblib", joblib_base[i]),
        ]:
            results.append(
                {
                    "engine": engine,
                    "sample_size": size,
                    "duration_secs": duration,
                    "records_per_second": round(size / duration, 2),
                    "peak_memory_mb": None,
                    "cpu_cores_used": 8 if engine != "pandas" else 1,
                    "benchmarked_at": "2024-01-08T02:50:00Z",
                }
            )
    return {
        "run_id": "run-20240108-020000-0007",
        "results": results,
        "spark_vs_pandas_speedup": round(1_000_000 / 95.0 / (1_000_000 / 11.2), 2),
        "spark_vs_joblib_speedup": round(1_000_000 / 95.0 / (1_000_000 / 30.1), 2),
    }


# ---------------------------------------------------------------------------
# Tab 1: Batch Run History
# ---------------------------------------------------------------------------


def load_batch_history() -> Tuple[pd.DataFrame, go.Figure, go.Figure, str]:
    data = _get("/batch-runs", params={"size": 20})
    stats = _get("/stats")

    if data:
        df = pd.DataFrame(data["items"])
    else:
        df = _demo_batch_runs()

    # Format for display
    display_df = df[
        [
            "run_id",
            "status",
            "started_at",
            "records_scored",
            "spark_duration_secs",
            "score_mean",
            "score_p50",
            "psi_vs_previous",
            "drift_flagged",
            "records_per_second",
        ]
    ].copy()
    display_df.columns = [
        "Run ID",
        "Status",
        "Started At",
        "Records Scored",
        "Spark Duration (s)",
        "Score Mean",
        "Score P50",
        "PSI vs Prev",
        "Drift Flagged",
        "Rec/sec",
    ]

    # Score mean trend
    completed = df[df["status"].isin(["validated", "completed"])].copy()
    score_fig = go.Figure()
    if not completed.empty and "score_mean" in completed.columns:
        score_fig.add_trace(
            go.Scatter(
                x=completed["started_at"].astype(str),
                y=completed["score_mean"].astype(float),
                mode="lines+markers",
                name="Score Mean",
                line=dict(color="#E25A1C", width=2),
                marker=dict(size=8),
            )
        )
        score_fig.add_trace(
            go.Scatter(
                x=completed["started_at"].astype(str),
                y=completed["score_p50"].astype(float),
                mode="lines+markers",
                name="Score P50",
                line=dict(color="#1f77b4", width=2, dash="dash"),
                marker=dict(size=8),
            )
        )
    score_fig.update_layout(
        title="Churn Score Statistics Across Runs",
        xaxis_title="Run Date",
        yaxis_title="Churn Probability",
        template="plotly_white",
        height=350,
        showlegend=True,
    )

    # PSI trend
    psi_fig = go.Figure()
    if not completed.empty and "psi_vs_previous" in completed.columns:
        psi_vals = completed["psi_vs_previous"].dropna()
        dates = completed.loc[psi_vals.index, "started_at"].astype(str)
        psi_fig.add_trace(
            go.Bar(
                x=dates,
                y=psi_vals.astype(float),
                name="PSI",
                marker_color=[
                    "#d62728" if v > 0.20 else "#ff7f0e" if v > 0.10 else "#2ca02c"
                    for v in psi_vals
                ],
            )
        )
        psi_fig.add_hline(
            y=0.20,
            line_dash="dash",
            line_color="#d62728",
            annotation_text="Drift threshold (0.20)",
        )
        psi_fig.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text="Monitor threshold (0.10)",
        )
    psi_fig.update_layout(
        title="Population Stability Index (PSI) — Score Distribution Drift",
        xaxis_title="Run Date",
        yaxis_title="PSI",
        template="plotly_white",
        height=350,
    )

    # Summary stats string
    if stats:
        summary = (
            f"**Total runs:** {stats.get('total_batch_runs', 'N/A')} | "
            f"**Total predictions:** {stats.get('total_predictions', 0):,} | "
            f"**Avg Spark duration:** {stats.get('avg_spark_duration_secs', 'N/A')}s | "
            f"**Avg throughput:** {stats.get('avg_records_per_second', 'N/A'):,.0f} rec/s"
            if stats.get("avg_records_per_second")
            else f"**Total runs:** {stats.get('total_batch_runs', 'N/A')}"
        )
    else:
        summary = "_Demo mode — FastAPI not reachable_"

    return display_df, score_fig, psi_fig, summary


# ---------------------------------------------------------------------------
# Tab 2: Score Distribution
# ---------------------------------------------------------------------------


def load_score_distribution(run_id: str) -> Tuple[go.Figure, pd.DataFrame, str]:
    if run_id.strip():
        data = _get(f"/batch-runs/{run_id.strip()}/distribution")
    else:
        data = None

    if not data:
        data = _demo_score_distribution()

    bins = data["bins"]
    labels = [f"{b['bin_lower']:.1f}–{b['bin_upper']:.1f}" for b in bins]
    counts = [b["count"] for b in bins]
    fractions = [b["fraction"] for b in bins]

    # Color by risk tier
    colors = []
    for b in bins:
        mid = (b["bin_lower"] + b["bin_upper"]) / 2
        if mid >= 0.70:
            colors.append("#d62728")  # high risk — red
        elif mid >= 0.40:
            colors.append("#ff7f0e")  # medium risk — orange
        else:
            colors.append("#2ca02c")  # low risk — green

    hist_fig = go.Figure()
    hist_fig.add_trace(
        go.Bar(
            x=labels,
            y=fractions,
            name="Score Distribution",
            marker_color=colors,
            text=[f"{c:,}" for c in counts],
            textposition="outside",
        )
    )
    hist_fig.update_layout(
        title=f"Churn Score Distribution — Run: {data['run_id'][:40]}",
        xaxis_title="Churn Probability Range",
        yaxis_title="Fraction of Customers",
        template="plotly_white",
        height=400,
        showlegend=False,
        xaxis_tickangle=-30,
        annotations=[
            dict(
                x=0.98,
                y=0.95,
                xref="paper",
                yref="paper",
                text="<b style='color:#2ca02c'>■ Low risk</b>  "
                "<b style='color:#ff7f0e'>■ Medium risk</b>  "
                "<b style='color:#d62728'>■ High risk</b>",
                showarrow=False,
                font=dict(size=11),
                align="right",
            ),
        ],
    )

    # Stats table
    total = sum(counts)
    risk_stats = {
        "Low (0.0–0.4)": sum(c for b, c in zip(bins, counts) if b["bin_upper"] <= 0.40),
        "Medium (0.4–0.7)": sum(
            c for b, c in zip(bins, counts) if 0.40 < b["bin_upper"] <= 0.70
        ),
        "High (0.7–1.0)": sum(c for b, c in zip(bins, counts) if b["bin_upper"] > 0.70),
    }
    stats_df = pd.DataFrame(
        [
            {"Risk Tier": tier, "Count": count, "Fraction": f"{count / total:.1%}"}
            for tier, count in risk_stats.items()
        ]
        + [{"Risk Tier": "Total", "Count": total, "Fraction": "100.0%"}]
    )

    note = f"Run ID: `{data['run_id']}`  |  Total records: {total:,}"

    return hist_fig, stats_df, note


# ---------------------------------------------------------------------------
# Tab 3: Benchmark
# ---------------------------------------------------------------------------


def load_benchmark_results() -> Tuple[go.Figure, go.Figure, pd.DataFrame, str]:
    data = _get("/benchmark")
    if not data:
        data = _demo_benchmark()

    results = data["results"]
    df = pd.DataFrame(results)

    ENGINE_COLORS = {
        "pyspark": "#E25A1C",
        "pandas": "#1f77b4",
        "joblib": "#2ca02c",
    }
    ENGINE_LABELS = {
        "pyspark": "PySpark (distributed)",
        "pandas": "pandas (single-threaded)",
        "joblib": "joblib Parallel (multi-process)",
    }

    # Throughput bar chart (at each sample size)
    bar_fig = go.Figure()
    for engine in ["pandas", "joblib", "pyspark"]:
        sub = df[df["engine"] == engine].sort_values("sample_size")
        bar_fig.add_trace(
            go.Bar(
                x=[f"{int(s):,}" for s in sub["sample_size"]],
                y=sub["records_per_second"],
                name=ENGINE_LABELS.get(engine, engine),
                marker_color=ENGINE_COLORS.get(engine, "#888"),
            )
        )
    bar_fig.update_layout(
        title="Inference Throughput by Engine and Sample Size",
        xaxis_title="Sample Size (rows)",
        yaxis_title="Records / Second",
        barmode="group",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Duration line chart (log-log)
    line_fig = go.Figure()
    for engine in ["pandas", "joblib", "pyspark"]:
        sub = df[df["engine"] == engine].sort_values("sample_size")
        line_fig.add_trace(
            go.Scatter(
                x=sub["sample_size"],
                y=sub["duration_secs"],
                name=ENGINE_LABELS.get(engine, engine),
                mode="lines+markers",
                line=dict(color=ENGINE_COLORS.get(engine, "#888"), width=2),
                marker=dict(size=9),
            )
        )
    line_fig.update_layout(
        title="Inference Duration vs Sample Size",
        xaxis_title="Sample Size (rows)",
        yaxis_title="Duration (seconds)",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    # Summary table
    max_size = df["sample_size"].max()
    summary_rows = []
    for engine in ["pyspark", "pandas", "joblib"]:
        sub = df[(df["engine"] == engine) & (df["sample_size"] == max_size)]
        if not sub.empty:
            row = sub.iloc[0]
            summary_rows.append(
                {
                    "Engine": ENGINE_LABELS.get(engine, engine),
                    "1M Duration (s)": f"{row['duration_secs']:.1f}",
                    "Throughput (rec/s)": f"{int(row['records_per_second']):,}",
                    "CPU Cores": int(row.get("cpu_cores_used", 0) or 0),
                }
            )
    summary_df = pd.DataFrame(summary_rows)

    speedup_text = ""
    if data.get("spark_vs_pandas_speedup"):
        s_vs_p = data["spark_vs_pandas_speedup"]
        s_vs_j = data.get("spark_vs_joblib_speedup", "N/A")
        direction = "faster" if s_vs_p > 1 else "slower"
        speedup_text = (
            f"**At 1M records:** PySpark is **{abs(s_vs_p):.1f}x {direction}** than pandas "
            f"and **{abs(s_vs_j):.1f}x {'faster' if s_vs_j > 1 else 'slower'}** than joblib.  \n"
            "Note: PySpark has a ~30-50s JVM startup cost. Below ~100K records, "
            "pandas is faster. Above 500K records, PySpark's parallelism dominates."
        )

    return bar_fig, line_fig, summary_df, speedup_text


# ---------------------------------------------------------------------------
# Tab 4: Customer Lookup
# ---------------------------------------------------------------------------


def lookup_customer(customer_id: str) -> Tuple[str, pd.DataFrame]:
    cid = customer_id.strip()
    if not cid:
        return "Enter a customer ID above.", pd.DataFrame()

    data = _get(f"/score/{cid}")
    if not data:
        return (
            f"No score found for `{cid}`. Run the batch pipeline first.",
            pd.DataFrame(),
        )

    prob = data["churn_probability"]
    risk = data["risk_tier"].upper()
    emoji = "🔴" if risk == "HIGH" else "🟡" if risk == "MEDIUM" else "🟢"

    summary = (
        f"### {emoji} {risk} RISK\n"
        f"**Customer ID:** `{data['customer_id']}`  \n"
        f"**Churn Probability:** `{prob:.1%}`  \n"
        f"**Churn Decile:** `{data['churn_decile']} / 10` "
        f"(top {data['churn_decile'] * 10}% most likely to churn)  \n"
        f"**Model Version:** `{data['model_version']}`  \n"
        f"**Scored At:** `{data['scored_at']}`  \n"
        f"**Run ID:** `{data['run_id']}`  \n"
    )

    details_df = pd.DataFrame(
        [
            {
                "Field": "Churn Probability",
                "Value": f"{prob:.4f}",
            },
            {
                "Field": "Churn Decile",
                "Value": f"{data['churn_decile']} of 10",
            },
            {
                "Field": "Risk Tier",
                "Value": risk,
            },
            {
                "Field": "Model Version",
                "Value": data["model_version"],
            },
            {
                "Field": "Run ID",
                "Value": data["run_id"],
            },
            {
                "Field": "Scored At",
                "Value": data["scored_at"],
            },
        ]
    )

    return summary, details_df


def lookup_bulk(customer_ids_text: str) -> pd.DataFrame:
    ids = [
        line.strip() for line in customer_ids_text.strip().splitlines() if line.strip()
    ]
    if not ids:
        return pd.DataFrame()
    if len(ids) > 20:
        ids = ids[:20]

    data = _post("/scores/bulk", {"customer_ids": ids})
    if not data or not data.get("found"):
        return pd.DataFrame({"customer_id": ids, "status": ["Not found"] * len(ids)})

    rows = []
    found_map = {r["customer_id"]: r for r in data["found"]}
    for cid in ids:
        if cid in found_map:
            r = found_map[cid]
            rows.append(
                {
                    "customer_id": r["customer_id"],
                    "churn_probability": f"{r['churn_probability']:.1%}",
                    "risk_tier": r["risk_tier"].upper(),
                    "decile": r["churn_decile"],
                    "model_version": r["model_version"],
                    "scored_at": r["scored_at"],
                }
            )
        else:
            rows.append(
                {
                    "customer_id": cid,
                    "churn_probability": "N/A",
                    "risk_tier": "NOT FOUND",
                    "decile": "N/A",
                    "model_version": "N/A",
                    "scored_at": "N/A",
                }
            )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Build Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title=settings.gradio.title,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # Batch Inference Monitoring Dashboard
            **ML System Design — Nightly Churn Scoring Pipeline**

            This dashboard monitors a production batch inference system that scores
            **1,000,000 customers** for churn probability every night using PySpark.
            The pipeline runs on Airflow (2:00 AM), writes predictions to PostgreSQL,
            and serves them via FastAPI — no model loaded at serve time.
            """
        )

        # ----------------------------------------------------------------
        # Tab 1: Batch Run History
        # ----------------------------------------------------------------
        with gr.Tab("Batch Run History"):
            gr.Markdown(
                "View all pipeline executions, their duration, throughput, "
                "and score statistics. PSI chart shows distribution drift over time."
            )
            with gr.Row():
                refresh_btn = gr.Button("Refresh", variant="primary", scale=0)
                summary_md = gr.Markdown("_Loading..._")

            runs_table = gr.DataFrame(
                label="Pipeline Executions",
                interactive=False,
                wrap=True,
            )
            with gr.Row():
                score_trend_chart = gr.Plot(label="Score Statistics Trend")
                psi_trend_chart = gr.Plot(label="PSI Drift Trend")

            def _refresh():
                df, sf, pf, summary = load_batch_history()
                return df, sf, pf, summary

            refresh_btn.click(
                _refresh,
                outputs=[runs_table, score_trend_chart, psi_trend_chart, summary_md],
            )
            demo.load(
                _refresh,
                outputs=[runs_table, score_trend_chart, psi_trend_chart, summary_md],
            )

        # ----------------------------------------------------------------
        # Tab 2: Score Distribution
        # ----------------------------------------------------------------
        with gr.Tab("Score Distribution"):
            gr.Markdown(
                "Explore the churn probability distribution for any batch run. "
                "Green = low risk, Orange = medium risk, Red = high risk."
            )
            with gr.Row():
                run_id_input = gr.Textbox(
                    label="Run ID (leave blank for latest demo)",
                    placeholder="run-20240101-020000-ab12",
                    scale=3,
                )
                dist_btn = gr.Button("Load Distribution", variant="primary", scale=0)

            dist_chart = gr.Plot(label="Score Histogram")
            with gr.Row():
                stats_table = gr.DataFrame(
                    label="Risk Tier Breakdown", interactive=False
                )
                dist_note = gr.Markdown()

            dist_btn.click(
                load_score_distribution,
                inputs=[run_id_input],
                outputs=[dist_chart, stats_table, dist_note],
            )
            demo.load(
                lambda: load_score_distribution(""),
                outputs=[dist_chart, stats_table, dist_note],
            )

        # ----------------------------------------------------------------
        # Tab 3: Benchmark
        # ----------------------------------------------------------------
        with gr.Tab("Benchmark Comparison"):
            gr.Markdown(
                """
                **3-way inference benchmark: PySpark vs pandas vs joblib Parallel**

                This shows *when* each engine wins:
                - **pandas** wins at small scale (<50K rows) — no JVM startup overhead
                - **joblib** is the middle ground — multi-process, no Spark complexity
                - **PySpark** wins at scale (500K+) — true distributed parallelism, cluster-scalable
                """
            )
            bench_btn = gr.Button("Load Benchmark Results", variant="primary")
            with gr.Row():
                throughput_chart = gr.Plot(label="Throughput (records/sec)")
                duration_chart = gr.Plot(label="Duration (log scale)")
            bench_table = gr.DataFrame(label="1M Record Summary", interactive=False)
            speedup_md = gr.Markdown()

            bench_btn.click(
                load_benchmark_results,
                outputs=[throughput_chart, duration_chart, bench_table, speedup_md],
            )
            demo.load(
                load_benchmark_results,
                outputs=[throughput_chart, duration_chart, bench_table, speedup_md],
            )

        # ----------------------------------------------------------------
        # Tab 4: Customer Lookup
        # ----------------------------------------------------------------
        with gr.Tab("Customer Lookup"):
            gr.Markdown(
                "Look up the latest churn score for any customer. "
                "Reads from the `v_latest_scores` PostgreSQL view — O(log n) indexed lookup, <10ms."
            )

            with gr.Tab("Single Customer"):
                with gr.Row():
                    cid_input = gr.Textbox(
                        label="Customer ID",
                        placeholder="CUST-0000-000042",
                        scale=3,
                    )
                    lookup_btn = gr.Button("Look Up", variant="primary", scale=0)
                score_summary = gr.Markdown()
                score_table = gr.DataFrame(label="Score Details", interactive=False)

                lookup_btn.click(
                    lookup_customer,
                    inputs=[cid_input],
                    outputs=[score_summary, score_table],
                )
                cid_input.submit(
                    lookup_customer,
                    inputs=[cid_input],
                    outputs=[score_summary, score_table],
                )

            with gr.Tab("Bulk Lookup (up to 20 IDs)"):
                bulk_input = gr.Textbox(
                    label="Customer IDs (one per line)",
                    placeholder="CUST-0000-000001\nCUST-0000-000002\n...",
                    lines=8,
                )
                bulk_btn = gr.Button("Look Up All", variant="primary")
                bulk_table = gr.DataFrame(label="Bulk Results", interactive=False)

                bulk_btn.click(
                    lookup_bulk,
                    inputs=[bulk_input],
                    outputs=[bulk_table],
                )

        gr.Markdown(
            """
            ---
            **Architecture:** PySpark `mapInPandas` → PostgreSQL → FastAPI → Gradio  
            **Model:** LightGBM churn classifier | **Schedule:** Airflow DAG, 2:00 AM daily  
            **Monitoring:** PSI (Population Stability Index) for score drift detection
            """
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=settings.gradio.host,
        server_port=settings.gradio.port,
        show_error=True,
        share=False,
    )
