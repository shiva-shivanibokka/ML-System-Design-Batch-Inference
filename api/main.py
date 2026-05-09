"""
Batch Inference Serving API
=============================
FastAPI application serving pre-computed churn scores from PostgreSQL.

Key design principle: NO model is loaded at serve time.
  The batch pipeline runs nightly, writes scores to PostgreSQL, and this API
  reads from an indexed view. Single-customer lookups return in <10ms.
  This is the "pre-compute and cache" pattern used at Airbnb, DoorDash, and
  LinkedIn for serving batch ML predictions at low latency.

Endpoints:
  GET  /health                          — liveness + DB check
  GET  /stats                           — aggregate pipeline statistics
  GET  /score/{customer_id}            — latest score for one customer
  POST /scores/bulk                    — batch lookup (up to 500 customers)
  GET  /batch-runs                     — paginated list of pipeline executions
  GET  /batch-runs/{run_id}            — single batch run details
  GET  /batch-runs/{run_id}/distribution — score histogram for a run
  GET  /batch-runs/latest              — most recent completed run
  GET  /benchmark                      — latest benchmark comparison results
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from configs.settings import settings
from db.connection import get_db, _async_engine, ping_database
from api.schemas import (
    BatchRunListResponse,
    BatchRunResponse,
    BenchmarkComparisonResponse,
    BenchmarkResultItem,
    BulkScoreRequest,
    BulkScoreResponse,
    CustomerScoreResponse,
    HealthResponse,
    ScoreDistributionResponse,
    ScoreHistogramBin,
    StatsResponse,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App lifespan — runs on startup and shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Batch Inference API starting...")
    # Test DB connectivity at startup — warn but don't crash
    if ping_database():
        logger.info("PostgreSQL: connected")
    else:
        logger.warning("PostgreSQL: not reachable — endpoints will return 503")
    yield
    logger.info("Batch Inference API shutting down.")
    await _async_engine.dispose()


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.api.title,
    version=settings.api.version,
    description=__doc__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in production
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware — request timing
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_timing_header(request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
    response.headers["X-Response-Time-Ms"] = str(elapsed_ms)
    return response


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Liveness and database connectivity check",
)
async def health(db: AsyncSession = Depends(get_db)):
    db_status = "connected"
    latest_run_id = None
    latest_run_status = None

    try:
        result = await db.execute(
            text("""
                SELECT run_id, status
                FROM batch_runs
                ORDER BY started_at DESC
                LIMIT 1
            """)
        )
        row = result.fetchone()
        if row:
            latest_run_id = row[0]
            latest_run_status = row[1]
    except Exception as e:
        db_status = "unreachable"
        logger.error(f"Health check DB error: {e}")

    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        latest_run_id=latest_run_id,
        latest_run_status=latest_run_status,
        api_version=settings.api.version,
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["System"],
    summary="Aggregate pipeline statistics",
)
async def stats(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
        SELECT
            (SELECT COUNT(*) FROM batch_runs)                      AS total_batch_runs,
            (SELECT COUNT(*) FROM predictions)                     AS total_predictions,
            (SELECT run_id FROM batch_runs
             WHERE status IN ('completed','validated')
             ORDER BY started_at DESC LIMIT 1)                     AS latest_run_id,
            (SELECT completed_at FROM batch_runs
             WHERE status IN ('completed','validated')
             ORDER BY started_at DESC LIMIT 1)                     AS latest_run_completed_at,
            (SELECT model_version FROM batch_runs
             ORDER BY started_at DESC LIMIT 1)                     AS latest_model_version,
            (SELECT AVG(spark_duration_secs)
             FROM batch_runs WHERE status = 'validated')           AS avg_spark_duration,
            (SELECT AVG(records_scored / NULLIF(spark_duration_secs,0))
             FROM batch_runs WHERE status = 'validated')           AS avg_records_per_sec
    """)
    )
    row = result.fetchone()
    if not row:
        return StatsResponse(total_batch_runs=0, total_predictions=0)

    return StatsResponse(
        total_batch_runs=int(row[0] or 0),
        total_predictions=int(row[1] or 0),
        latest_run_id=row[2],
        latest_run_completed_at=row[3],
        latest_model_version=row[4],
        avg_spark_duration_secs=round(float(row[5]), 2) if row[5] else None,
        avg_records_per_second=round(float(row[6]), 0) if row[6] else None,
    )


# ---------------------------------------------------------------------------
# Customer score lookup
# ---------------------------------------------------------------------------


@app.get(
    "/score/{customer_id}",
    response_model=CustomerScoreResponse,
    tags=["Scores"],
    summary="Get latest churn score for a customer",
    description=(
        "Returns the most recent churn prediction for the given customer_id. "
        "Reads from the indexed `v_latest_scores` view — O(log n) lookup, "
        "typically < 10ms."
    ),
)
async def get_customer_score(
    customer_id: str,
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        text("""
            SELECT
                customer_id, run_id, model_version, churn_probability,
                churn_label, churn_decile, risk_tier, scored_at
            FROM v_latest_scores
            WHERE customer_id = :cid
        """),
        {"cid": customer_id},
    )
    row = result.fetchone()

    if not row:
        raise HTTPException(
            status_code=404,
            detail=f"No score found for customer_id='{customer_id}'. "
            "Run the batch pipeline first.",
        )

    return CustomerScoreResponse(
        customer_id=row[0],
        run_id=row[1],
        model_version=row[2],
        churn_probability=float(row[3]),
        churn_label=bool(row[4]),
        churn_decile=int(row[5]),
        risk_tier=row[6],
        scored_at=row[7],
    )


@app.post(
    "/scores/bulk",
    response_model=BulkScoreResponse,
    tags=["Scores"],
    summary="Bulk lookup — up to 500 customer scores in one call",
)
async def get_bulk_scores(
    request: BulkScoreRequest,
    db: AsyncSession = Depends(get_db),
):
    ids = request.customer_ids

    result = await db.execute(
        text("""
            SELECT
                customer_id, run_id, model_version, churn_probability,
                churn_label, churn_decile, risk_tier, scored_at
            FROM v_latest_scores
            WHERE customer_id = ANY(:ids)
        """),
        {"ids": ids},
    )
    rows = result.fetchall()

    found_map = {
        row[0]: CustomerScoreResponse(
            customer_id=row[0],
            run_id=row[1],
            model_version=row[2],
            churn_probability=float(row[3]),
            churn_label=bool(row[4]),
            churn_decile=int(row[5]),
            risk_tier=row[6],
            scored_at=row[7],
        )
        for row in rows
    }

    found = [found_map[cid] for cid in ids if cid in found_map]
    not_found = [cid for cid in ids if cid not in found_map]

    return BulkScoreResponse(
        found=found,
        not_found=not_found,
        total_requested=len(ids),
        total_found=len(found),
    )


# ---------------------------------------------------------------------------
# Batch run history
# ---------------------------------------------------------------------------


@app.get(
    "/batch-runs",
    response_model=BatchRunListResponse,
    tags=["Batch Runs"],
    summary="Paginated list of all batch pipeline executions",
)
async def list_batch_runs(
    page: int = Query(default=1, ge=1),
    size: int = Query(
        default=settings.api.default_page_size, ge=1, le=settings.api.max_page_size
    ),
    status: Optional[str] = Query(default=None, description="Filter by status"),
    db: AsyncSession = Depends(get_db),
):
    offset = (page - 1) * size

    count_q = "SELECT COUNT(*) FROM batch_runs"
    runs_q = """
        SELECT * FROM v_batch_run_summary
        {where}
        LIMIT :size OFFSET :offset
    """

    params: dict = {"size": size, "offset": offset}
    where = ""
    if status:
        where = "WHERE status = :status"
        params["status"] = status
        count_q += " WHERE status = :status"

    total_result = await db.execute(text(count_q), params)
    total = total_result.scalar() or 0

    rows_result = await db.execute(text(runs_q.format(where=where)), params)
    rows = rows_result.fetchall()
    cols = rows_result.keys()

    items = [BatchRunResponse(**dict(zip(cols, row))) for row in rows]

    return BatchRunListResponse(total=total, page=page, size=size, items=items)


@app.get(
    "/batch-runs/latest",
    response_model=BatchRunResponse,
    tags=["Batch Runs"],
    summary="Most recent completed batch run",
)
async def get_latest_batch_run(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
        SELECT * FROM v_batch_run_summary
        WHERE status IN ('completed', 'validated')
        LIMIT 1
    """)
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="No completed batch runs found.")
    cols = result.keys()
    return BatchRunResponse(**dict(zip(cols, row)))


@app.get(
    "/batch-runs/{run_id}",
    response_model=BatchRunResponse,
    tags=["Batch Runs"],
    summary="Details for a specific batch run",
)
async def get_batch_run(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("SELECT * FROM v_batch_run_summary WHERE run_id = :run_id"),
        {"run_id": run_id},
    )
    row = result.fetchone()
    if not row:
        raise HTTPException(status_code=404, detail=f"Batch run '{run_id}' not found.")
    cols = result.keys()
    return BatchRunResponse(**dict(zip(cols, row)))


# ---------------------------------------------------------------------------
# Score distribution
# ---------------------------------------------------------------------------


@app.get(
    "/batch-runs/{run_id}/distribution",
    response_model=ScoreDistributionResponse,
    tags=["Batch Runs"],
    summary="Score histogram for a specific batch run (10 equal-width bins)",
)
async def get_score_distribution(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        text("""
            SELECT bin, bin_lower, bin_upper, count, fraction
            FROM v_score_histogram
            WHERE run_id = :run_id
            ORDER BY bin
        """),
        {"run_id": run_id},
    )
    rows = result.fetchall()

    if not rows:
        raise HTTPException(
            status_code=404,
            detail=f"No score data found for run_id='{run_id}'.",
        )

    bins = [
        ScoreHistogramBin(
            bin=int(row[0]),
            bin_lower=float(row[1]),
            bin_upper=float(row[2]),
            count=int(row[3]),
            fraction=float(row[4]),
        )
        for row in rows
    ]

    return ScoreDistributionResponse(run_id=run_id, n_bins=len(bins), bins=bins)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


@app.get(
    "/benchmark",
    response_model=BenchmarkComparisonResponse,
    tags=["Benchmark"],
    summary="Latest 3-way benchmark: PySpark vs pandas vs joblib",
)
async def get_benchmark_results(
    run_id: Optional[str] = Query(
        default=None, description="Specific run_id, or latest if omitted"
    ),
    db: AsyncSession = Depends(get_db),
):
    if run_id:
        q = "SELECT * FROM benchmark_results WHERE run_id = :run_id ORDER BY engine, sample_size"
        params = {"run_id": run_id}
    else:
        # Latest run_id that has benchmark data
        q = """
            SELECT * FROM benchmark_results
            WHERE run_id = (SELECT run_id FROM benchmark_results ORDER BY benchmarked_at DESC LIMIT 1)
            ORDER BY engine, sample_size
        """
        params = {}

    result = await db.execute(text(q), params)
    rows = result.fetchall()

    if not rows:
        raise HTTPException(status_code=404, detail="No benchmark results found.")

    cols = result.keys()
    items = [
        BenchmarkResultItem(
            **{k: v for k, v in zip(cols, row) if k in BenchmarkResultItem.model_fields}
        )
        for row in rows
    ]

    # Compute speedup: PySpark vs pandas and joblib at max sample size
    max_size = max(r.sample_size for r in items)
    rates = {r.engine: r.records_per_second for r in items if r.sample_size == max_size}
    spark_rps = rates.get("pyspark")
    pandas_rps = rates.get("pandas")
    joblib_rps = rates.get("joblib")

    return BenchmarkComparisonResponse(
        run_id=rows[0][1],  # run_id column
        results=items,
        spark_vs_pandas_speedup=(
            round(spark_rps / pandas_rps, 2) if spark_rps and pandas_rps else None
        ),
        spark_vs_joblib_speedup=(
            round(spark_rps / joblib_rps, 2) if spark_rps and joblib_rps else None
        ),
    )


# ---------------------------------------------------------------------------
# Entry point (for running directly without uvicorn CLI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        workers=2,
    )
