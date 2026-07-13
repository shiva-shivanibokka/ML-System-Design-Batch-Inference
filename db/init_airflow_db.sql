-- Give Airflow its own metadata database so its ~30 internal tables never mix
-- with the application schema (customers, predictions, batch_runs, benchmarks).
-- Runs once, at first container init (empty data dir only).
CREATE DATABASE airflow_meta;
