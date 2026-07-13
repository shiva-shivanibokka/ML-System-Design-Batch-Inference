"""
Typed settings loader.

Reads config.yaml once at import time. All environment variable overrides
are applied here so the rest of the codebase never calls os.getenv directly.

Usage:
    from configs.settings import settings
    settings.database.host
    settings.spark.n_partitions
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import yaml

# ---------------------------------------------------------------------------
# Locate config.yaml relative to this file's parent directory
# ---------------------------------------------------------------------------
_CONFIG_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml() -> dict:
    with open(_CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Dataclass hierarchy — mirrors config.yaml structure exactly
# ---------------------------------------------------------------------------


@dataclass
class DatabaseSettings:
    host: str
    port: int
    name: str
    user: str
    password: str
    pool_size: int
    max_overflow: int
    # Full connection string (Neon/Vercel style). When set, it wins over the parts.
    url_override: Optional[str] = None

    def _with_driver(self, driver: str) -> str:
        """Normalise url_override to the requested SQLAlchemy driver."""
        raw = self.url_override or ""
        # `postgres://` (Heroku/Neon style) → `postgresql://`
        raw = re.sub(r"^postgres(ql)?(\+\w+)?://", "postgresql://", raw, count=1)
        return raw.replace("postgresql://", f"postgresql+{driver}://", 1)

    @property
    def url(self) -> str:
        if self.url_override:
            return self._with_driver("psycopg2")  # sslmode=... kept (psycopg2 understands it)
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )

    @property
    def async_url(self) -> str:
        if self.url_override:
            # asyncpg doesn't accept the libpq `sslmode` query param — strip it;
            # SSL is supplied via connect_args in db/connection.py instead.
            return re.sub(r"[?&]sslmode=[^&]+", "", self._with_driver("asyncpg"))
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.name}"
        )


@dataclass
class DataSettings:
    dataset: str
    raw_dir: str
    output_path: str
    reference_date: int
    members_file: str
    transactions_file: str
    labels_file: str
    numeric_features: List[str]
    categorical_features: List[str]


@dataclass
class LGBMSettings:
    n_estimators: int
    learning_rate: float
    num_leaves: int
    max_depth: int
    min_child_samples: int
    subsample: float
    colsample_bytree: float
    reg_alpha: float
    reg_lambda: float
    class_weight: str
    random_state: int
    n_jobs: int


@dataclass
class ModelSettings:
    path: str
    version: str
    label_encoders_path: str
    feature_columns_path: str
    lgbm: LGBMSettings
    train_size: float
    val_size: float
    test_size: float


@dataclass
class SparkSettings:
    app_name: str
    master: str
    executor_memory: str
    driver_memory: str
    n_partitions: int
    input_path: str
    output_path: str
    log_level: str


@dataclass
class BenchmarkSettings:
    sample_sizes: List[int]
    n_joblib_workers: int
    joblib_backend: str
    results_path: str
    chart_path: str
    warmup_rows: int


@dataclass
class PipelineSettings:
    batch_run_id_prefix: str
    psi_threshold: float
    psi_n_bins: int
    min_records_expected: int
    max_null_score_pct: float
    score_range: List[float]


@dataclass
class APISettings:
    host: str
    port: int
    title: str
    version: str
    cache_ttl_seconds: int
    default_page_size: int
    max_page_size: int


@dataclass
class GradioSettings:
    host: str
    port: int
    title: str
    theme: str
    api_base_url: str


@dataclass
class AirflowSettings:
    dag_id: str
    schedule: str
    catchup: bool
    retries: int
    retry_delay_minutes: int
    sla_minutes: int
    email_on_failure: bool
    tags: List[str]


@dataclass
class Settings:
    database: DatabaseSettings
    data: DataSettings
    model: ModelSettings
    spark: SparkSettings
    benchmark: BenchmarkSettings
    pipeline: PipelineSettings
    api: APISettings
    gradio: GradioSettings
    airflow: AirflowSettings


# ---------------------------------------------------------------------------
# Builder — applies environment variable overrides
# ---------------------------------------------------------------------------


def _build_settings() -> Settings:
    raw = _load_yaml()

    db_raw = raw["database"]
    db = DatabaseSettings(
        host=os.getenv("POSTGRES_HOST", db_raw["host"]),
        port=int(os.getenv("POSTGRES_PORT", db_raw["port"])),
        name=os.getenv("POSTGRES_DB", db_raw["name"]),
        user=os.getenv("POSTGRES_USER", db_raw["user"]),
        password=os.getenv("POSTGRES_PASSWORD", db_raw["password"]),
        pool_size=db_raw["pool_size"],
        max_overflow=db_raw["max_overflow"],
        url_override=os.getenv("DATABASE_URL"),
    )

    data_raw = raw["data"]
    data = DataSettings(
        dataset=data_raw["dataset"],
        raw_dir=data_raw["raw_dir"],
        output_path=data_raw["output_path"],
        reference_date=int(data_raw["reference_date"]),
        members_file=data_raw["members_file"],
        transactions_file=data_raw["transactions_file"],
        labels_file=data_raw["labels_file"],
        numeric_features=data_raw["numeric_features"],
        categorical_features=data_raw["categorical_features"],
    )

    model_raw = raw["model"]
    lgbm_raw = model_raw["lgbm"]
    model = ModelSettings(
        path=model_raw["path"],
        version=model_raw["version"],
        label_encoders_path=model_raw["label_encoders_path"],
        feature_columns_path=model_raw["feature_columns_path"],
        lgbm=LGBMSettings(**lgbm_raw),
        train_size=model_raw["train_size"],
        val_size=model_raw["val_size"],
        test_size=model_raw["test_size"],
    )

    spark_raw = raw["spark"]
    spark = SparkSettings(
        app_name=spark_raw["app_name"],
        master=os.getenv("SPARK_MASTER", spark_raw["master"]),
        executor_memory=spark_raw["executor_memory"],
        driver_memory=spark_raw["driver_memory"],
        n_partitions=spark_raw["n_partitions"],
        input_path=spark_raw["input_path"],
        output_path=spark_raw["output_path"],
        log_level=spark_raw["log_level"],
    )

    bench_raw = raw["benchmark"]
    benchmark = BenchmarkSettings(
        sample_sizes=bench_raw["sample_sizes"],
        n_joblib_workers=bench_raw["n_joblib_workers"],
        joblib_backend=bench_raw["joblib_backend"],
        results_path=bench_raw["results_path"],
        chart_path=bench_raw["chart_path"],
        warmup_rows=bench_raw["warmup_rows"],
    )

    pipeline_raw = raw["pipeline"]
    pipeline = PipelineSettings(
        batch_run_id_prefix=pipeline_raw["batch_run_id_prefix"],
        psi_threshold=pipeline_raw["psi_threshold"],
        psi_n_bins=pipeline_raw["psi_n_bins"],
        min_records_expected=pipeline_raw["min_records_expected"],
        max_null_score_pct=pipeline_raw["max_null_score_pct"],
        score_range=pipeline_raw["score_range"],
    )

    api_raw = raw["api"]
    api = APISettings(
        host=api_raw["host"],
        port=int(os.getenv("API_PORT", api_raw["port"])),
        title=api_raw["title"],
        version=api_raw["version"],
        cache_ttl_seconds=api_raw["cache_ttl_seconds"],
        default_page_size=api_raw["default_page_size"],
        max_page_size=api_raw["max_page_size"],
    )

    gradio_raw = raw["gradio"]
    gradio = GradioSettings(
        host=gradio_raw["host"],
        port=int(os.getenv("GRADIO_PORT", gradio_raw["port"])),
        title=gradio_raw["title"],
        theme=gradio_raw["theme"],
        api_base_url=os.getenv("API_BASE_URL", gradio_raw["api_base_url"]),
    )

    airflow_raw = raw["airflow"]
    airflow = AirflowSettings(
        dag_id=airflow_raw["dag_id"],
        schedule=airflow_raw["schedule"],
        catchup=airflow_raw["catchup"],
        retries=airflow_raw["retries"],
        retry_delay_minutes=airflow_raw["retry_delay_minutes"],
        sla_minutes=airflow_raw["sla_minutes"],
        email_on_failure=airflow_raw["email_on_failure"],
        tags=airflow_raw["tags"],
    )

    return Settings(
        database=db,
        data=data,
        model=model,
        spark=spark,
        benchmark=benchmark,
        pipeline=pipeline,
        api=api,
        gradio=gradio,
        airflow=airflow,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
settings = _build_settings()
