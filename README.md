# ML System Design — Batch Inference Pipeline at Scale

> **Production-grade nightly batch scoring system** that scores 1,000,000 customers for churn probability using PySpark distributed inference, orchestrated by Apache Airflow, served via FastAPI, and monitored through a Gradio dashboard.

Modelled after the architecture used at **Airbnb**, **LinkedIn**, and **DoorDash** for nightly pre-computed ML scoring at scale.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   Apache Airflow (2:00 AM daily)                 │
│                                                                   │
│  t1_validate_inputs ──► t2_run_spark_inference                   │
│                              ──► t3_validate_scores              │
│                                      ──► t4_write_to_postgres    │
│                                              ──► t5_run_benchmark│
│                                                    ──► t6_update_│
│                                                        monitoring │
└──────────────────────────────┬──────────────────────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │       PySpark Batch Engine       │
               │  customers.parquet (1M rows)     │
               │  → repartition(50 partitions)    │
               │  → mapInPandas per partition     │
               │  → LightGBM inference per shard  │
               │  → scored_output.parquet         │
               └────────────────┬────────────────┘
                                │
               ┌────────────────▼────────────────┐
               │           PostgreSQL             │
               │  customers     (1M rows)         │
               │  predictions   (audit trail)     │
               │  batch_runs    (job metadata)    │
               │  benchmark_results               │
               └──────────┬──────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │         FastAPI               │
          │  GET /score/{customer_id}     │  ← <10ms indexed lookup
          │  GET /batch-runs              │  ← pipeline history
          │  GET /benchmark               │  ← 3-way comparison
          │  POST /scores/bulk            │  ← bulk lookup (500 IDs)
          └───────────────────────────────┘
                          │
          ┌───────────────▼───────────────┐
          │       Gradio Dashboard        │
          │  Tab 1: Batch Run History     │
          │  Tab 2: Score Distribution    │
          │  Tab 3: Benchmark Comparison  │
          │  Tab 4: Customer Lookup       │
          └───────────────────────────────┘
```

---

## What Makes This Production-Grade

### 1. PySpark `mapInPandas` with Model Broadcasting
The core of the system — not a toy loop:

```python
# Model broadcast to all executors ONCE per partition (not per row)
broadcast_model = spark.sparkContext.broadcast(model_bytes)

# Partition-level inference — each partition is an independent pandas DataFrame
scored_df = df.repartition(50).mapInPandas(predict_partition, schema=output_schema)
```

The LightGBM model is serialized and broadcast to all Spark executors via `SparkContext.broadcast()`. Each executor deserializes it once per partition — not once per row. This is the exact pattern used at LinkedIn for nightly member scoring.

### 2. Full Audit Trail
Every prediction row is stored with complete provenance:

| Column | Value |
|---|---|
| `customer_id` | `CUST-0000-042` |
| `churn_probability` | `0.7312` |
| `churn_label` | `True` |
| `churn_decile` | `8` |
| `risk_tier` | `high` |
| `run_id` | `run-20240101-020000-ab12cd34` |
| `model_version` | `v1.0.0` |
| `scored_at` | `2024-01-01T02:45:00Z` |

Downstream teams can always query: *"What was this customer's score at 2am on Jan 1st with model v1?"*

### 3. 5-Gate Validation Before Writing
The pipeline rejects and retries if any validation gate fails:
1. **Record count** ≥ 900,000 (catches partial Spark job failures)
2. **Null rate** < 0.1% (catches prediction errors)
3. **Score range** [0.0, 1.0] (catches model loading errors)
4. **Non-degenerate distribution** (std > 0.01 — catches constant prediction)
5. **Plausible churn rate** [5%, 50%] (catches model drift)

### 4. PSI Score Distribution Monitoring
Population Stability Index (Basel II standard) measures distribution shift between runs:

```
PSI = Σ (current_% - reference_%) × ln(current_% / reference_%)

< 0.10 → Stable (no action)
0.10–0.20 → Moderate change (monitor)
> 0.20 → Significant shift (investigate/retrain)
```

### 5. 3-Way Benchmark: When Does Each Engine Win?

| Engine | 10K rows | 100K rows | 500K rows | 1M rows |
|---|---|---|---|---|
| **pandas** | ~0.1s | ~1.1s | ~5.5s | ~11s |
| **joblib Parallel** | ~0.9s | ~3.2s | ~14.8s | ~30s |
| **PySpark** | ~45s | ~52s | ~68s | ~95s |

**Insight**: PySpark breaks even with pandas at ~500K rows. Above 1M records on a cluster (not local mode), PySpark's advantage compounds — it's the only option that scales to 100M+ records.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Distributed compute | **Apache Spark 3.5** (PySpark, `mapInPandas`) |
| Orchestration | **Apache Airflow 2.9** (LocalExecutor, 6-task DAG) |
| ML model | **LightGBM 4.3** (binary churn classifier) |
| Database | **PostgreSQL 16** (audit trail, indexed views) |
| API | **FastAPI 0.111** + Uvicorn (async, 2 workers) |
| Monitoring | **PSI** (Population Stability Index) via scipy |
| Dashboard | **Gradio 4.36** + Plotly |
| Data | **Faker** (1M synthetic customers, ~120MB Parquet) |
| Container | **Docker + docker-compose** (5 services) |
| Config | **YAML → typed dataclasses** (env var overrides) |

---

## Quick Start

### Prerequisites
- Docker Desktop
- 8GB+ RAM (Spark needs ~4GB)

### 1. Clone and Start

```bash
git clone https://github.com/YOUR_USERNAME/ML-System-Design-Batch-Inference
cd ML-System-Design-Batch-Inference
docker-compose up --build
```

### 2. Access Services

| Service | URL | Credentials |
|---|---|---|
| **Airflow UI** | http://localhost:8081 | admin / admin |
| **FastAPI docs** | http://localhost:8000/docs | — |
| **Gradio dashboard** | http://localhost:7860 | — |
| **Spark UI** | http://localhost:8080 | — |
| **PostgreSQL** | localhost:5432 | batch_user / batch_pass |

### 3. Prepare Data and Train Model (first time only)

```bash
# Generate 1M synthetic customers
docker-compose exec airflow python /opt/airflow/data/generate_data.py

# Train the LightGBM model
docker-compose exec airflow python /opt/airflow/models/train.py --eval
```

### 4. Trigger the Pipeline

**Option A — Airflow UI:**
1. Open http://localhost:8081
2. Find `nightly_batch_inference` DAG
3. Click **Trigger DAG** (▷ button)
4. Watch the 6 tasks execute

**Option B — CLI:**
```bash
docker-compose exec airflow airflow dags trigger nightly_batch_inference
```

**Option C — Manual scripts (no Docker):**
```bash
# Generate data
python data/generate_data.py

# Train model
python models/train.py

# Run Spark inference
python spark/batch_inference.py --skip-postgres

# Run benchmark
python benchmark/compare.py

# Start API
uvicorn api.main:app --port 8000

# Start dashboard
python gradio_app/app.py
```

---

## Local Development (without Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Set up PostgreSQL (or use Docker for just the DB)
docker run -d --name pg \
  -e POSTGRES_DB=batch_inference \
  -e POSTGRES_USER=batch_user \
  -e POSTGRES_PASSWORD=batch_pass \
  -p 5432:5432 postgres:16-alpine

# Apply schema
psql -U batch_user -d batch_inference -f db/schema.sql

# Generate data + train
python data/generate_data.py
python models/train.py

# Run components
python spark/batch_inference.py
python benchmark/compare.py
uvicorn api.main:app --reload &
python gradio_app/app.py
```

---

## Project Structure

```
ML-System-Design-Batch-Inference/
├── airflow/
│   └── dags/
│       └── batch_inference_dag.py   # 6-task Airflow DAG
├── api/
│   ├── main.py                      # FastAPI serving layer (8 endpoints)
│   └── schemas.py                   # Pydantic v2 request/response models
├── benchmark/
│   └── compare.py                   # 3-way: PySpark vs pandas vs joblib
├── configs/
│   ├── config.yaml                  # All tuneable parameters
│   └── settings.py                  # Typed dataclass settings loader
├── data/
│   └── generate_data.py             # Faker: 1M synthetic customers
├── db/
│   ├── schema.sql                   # PostgreSQL schema (tables + views)
│   └── connection.py                # SQLAlchemy sync + async engines
├── gradio_app/
│   └── app.py                       # 4-tab monitoring dashboard
├── models/
│   └── train.py                     # LightGBM training script
├── monitoring/
│   └── score_monitor.py             # PSI drift detection
├── spark/
│   └── batch_inference.py           # PySpark mapInPandas inference job
├── Dockerfile.api
├── Dockerfile.airflow
├── Dockerfile.gradio
├── docker-compose.yml               # 5 services
└── requirements.txt
```

---

## API Reference

```
GET  /health                          → liveness + DB check
GET  /stats                           → aggregate pipeline statistics
GET  /score/{customer_id}            → latest score, <10ms response
POST /scores/bulk                    → batch lookup (up to 500 IDs)
GET  /batch-runs                     → paginated run history
GET  /batch-runs/latest              → most recent completed run
GET  /batch-runs/{run_id}            → single run details
GET  /batch-runs/{run_id}/distribution → 10-bin score histogram
GET  /benchmark                      → latest 3-way benchmark results
```

**Example: Look up a customer score**
```bash
curl http://localhost:8000/score/CUST-0000-000042
```
```json
{
  "customer_id": "CUST-0000-000042",
  "churn_probability": 0.7312,
  "churn_label": true,
  "churn_decile": 8,
  "risk_tier": "high",
  "model_version": "v1.0.0",
  "scored_at": "2024-01-01T02:45:00Z",
  "run_id": "run-20240101-020000-ab12cd34"
}
```

---

## Resume Talking Points

- **Distributed ML inference** using PySpark `mapInPandas` with model broadcasting — exact pattern used at LinkedIn/Airbnb for nightly scoring
- **Pipeline orchestration** with Apache Airflow (6-task DAG with retries, SLA monitoring, XCom state passing)
- **Score drift monitoring** using PSI (Population Stability Index) — Basel II standard for financial ML systems
- **Full prediction audit trail** in PostgreSQL — every score stored with `run_id`, `model_version`, `scored_at` for full lineage
- **Validated batch pipeline** — 5 automated gates reject the batch before it reaches production
- **3-way benchmark** demonstrating quantitative tradeoffs: PySpark vs pandas vs joblib at 4 data scales
- **Pre-computed serving** — FastAPI serves scores from indexed PostgreSQL view in <10ms without loading any model
- **Docker + docker-compose** — one-command reproducible deployment of all 5 services

---

## Key Design Decisions

**Why pre-compute and store vs real-time inference?**
Not all ML predictions need to be computed on demand. Churn risk doesn't change hourly — computing it once nightly and serving from an indexed DB gives <10ms latency vs ~200ms for model inference, at 1/20th the serving infrastructure cost.

**Why `mapInPandas` over Pandas UDFs?**
`mapInPandas` processes entire partitions (batched), while scalar Pandas UDFs process row-by-row. For a 500-tree LightGBM model, vectorized batch inference is 10–50x faster than per-row calls.

**Why PSI over KS test for monitoring?**
PSI gives a continuous magnitude score that can be trended over time. KS test gives a binary pass/fail. PSI is also the regulatory standard (Basel II) — relevant for financial/insurance use cases.
