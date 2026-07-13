"""
Pydantic v2 request/response schemas for the Batch Inference Serving API.
"""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Customer score lookup
# ---------------------------------------------------------------------------


class CustomerScoreResponse(BaseModel):
    """Latest churn score for a single customer."""

    customer_id: str
    run_id: str
    model_version: str
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_label: bool
    churn_decile: int = Field(..., ge=1, le=10)
    risk_tier: str = Field(..., pattern="^(low|medium|high)$")
    scored_at: datetime

    model_config = {
        "protected_namespaces": (),  # allow the `model_version` field name
        "json_schema_extra": {
            "example": {
                "customer_id": "CUST-0000-000042",
                "run_id": "run-20240101-020000-ab12cd34",
                "model_version": "v1.0.0",
                "churn_probability": 0.7312,
                "churn_label": True,
                "churn_decile": 8,
                "risk_tier": "high",
                "scored_at": "2024-01-01T02:45:00Z",
            }
        }
    }


class CustomerScoreNotFound(BaseModel):
    detail: str
    customer_id: str


# ---------------------------------------------------------------------------
# Batch run metadata
# ---------------------------------------------------------------------------


class BatchRunResponse(BaseModel):
    """Summary of a single batch pipeline execution."""

    run_id: str
    model_version: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    status: str
    records_read: Optional[int] = None
    records_scored: Optional[int] = None
    records_failed: Optional[int] = None
    spark_duration_secs: Optional[float] = None
    total_duration_secs: Optional[float] = None
    score_mean: Optional[float] = None
    score_p50: Optional[float] = None
    psi_vs_previous: Optional[float] = None
    drift_flagged: Optional[bool] = None
    validation_passed: Optional[bool] = None
    records_per_second: Optional[float] = None
    wall_clock_secs: Optional[float] = None

    model_config = {
        "protected_namespaces": (),  # allow the `model_version` field name
        "json_schema_extra": {
            "example": {
                "run_id": "run-20240101-020000-ab12cd34",
                "model_version": "v1.0.0",
                "started_at": "2024-01-01T02:00:00Z",
                "completed_at": "2024-01-01T02:47:30Z",
                "status": "validated",
                "records_scored": 1_000_000,
                "spark_duration_secs": 183.4,
                "score_mean": 0.1847,
                "score_p50": 0.1123,
                "psi_vs_previous": 0.0312,
                "drift_flagged": False,
                "validation_passed": True,
                "records_per_second": 5456.0,
            }
        }
    }


class BatchRunListResponse(BaseModel):
    total: int
    page: int
    size: int
    items: List[BatchRunResponse]


# ---------------------------------------------------------------------------
# Score distribution (histogram)
# ---------------------------------------------------------------------------


class ScoreHistogramBin(BaseModel):
    bin: int
    bin_lower: float
    bin_upper: float
    count: int
    fraction: float


class ScoreDistributionResponse(BaseModel):
    run_id: str
    n_bins: int
    bins: List[ScoreHistogramBin]


# ---------------------------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------------------------


class BenchmarkResultItem(BaseModel):
    engine: str
    sample_size: int
    duration_secs: float
    records_per_second: float
    peak_memory_mb: Optional[float] = None
    cpu_cores_used: Optional[int] = None
    benchmarked_at: datetime


class BenchmarkComparisonResponse(BaseModel):
    """3-way comparison: PySpark vs pandas vs joblib."""

    run_id: str
    results: List[BenchmarkResultItem]
    # Derived speedup metrics (PySpark vs others at max sample size)
    spark_vs_pandas_speedup: Optional[float] = None
    spark_vs_joblib_speedup: Optional[float] = None


# ---------------------------------------------------------------------------
# Health + system
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str  # "ok" or "degraded"
    database: str  # "connected" or "unreachable"
    latest_run_id: Optional[str] = None
    latest_run_status: Optional[str] = None
    api_version: str


class StatsResponse(BaseModel):
    total_batch_runs: int
    total_predictions: int
    latest_run_id: Optional[str] = None
    latest_run_completed_at: Optional[datetime] = None
    latest_model_version: Optional[str] = None
    avg_spark_duration_secs: Optional[float] = None
    avg_records_per_second: Optional[float] = None


# ---------------------------------------------------------------------------
# Bulk score lookup
# ---------------------------------------------------------------------------


class BulkScoreRequest(BaseModel):
    customer_ids: List[str] = Field(..., min_length=1, max_length=500)

    @field_validator("customer_ids")
    @classmethod
    def deduplicate(cls, v):
        return list(dict.fromkeys(v))  # deduplicate, preserve order


class BulkScoreResponse(BaseModel):
    found: List[CustomerScoreResponse]
    not_found: List[str]
    total_requested: int
    total_found: int
