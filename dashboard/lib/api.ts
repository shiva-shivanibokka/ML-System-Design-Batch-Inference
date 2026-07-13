// Typed client for the FastAPI serving layer.
// Base URL comes from NEXT_PUBLIC_API_URL (the API's Vercel deployment).
// Falls back to localhost for `npm run dev` against a local uvicorn.

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "http://localhost:8000";

export type CustomerScore = {
  customer_id: string;
  run_id: string;
  model_version: string;
  churn_probability: number;
  churn_label: boolean;
  churn_decile: number;
  risk_tier: "low" | "medium" | "high";
  scored_at: string;
};

export type BatchRun = {
  run_id: string;
  model_version: string;
  started_at: string;
  completed_at: string | null;
  status: string;
  records_scored: number | null;
  spark_duration_secs: number | null;
  score_mean: number | null;
  score_p50: number | null;
  psi_vs_previous: number | null;
  drift_flagged: boolean | null;
  validation_passed: boolean | null;
  records_per_second: number | null;
};

export type Stats = {
  total_batch_runs: number;
  total_predictions: number;
  latest_run_id: string | null;
  latest_model_version: string | null;
  avg_spark_duration_secs: number | null;
  avg_records_per_second: number | null;
};

export type HistBin = {
  bin: number;
  bin_lower: number;
  bin_upper: number;
  count: number;
  fraction: number;
};

export type ScoreDistribution = { run_id: string; n_bins: number; bins: HistBin[] };

export type BenchmarkItem = {
  engine: "pyspark" | "pandas" | "joblib";
  sample_size: number;
  duration_secs: number;
  records_per_second: number;
  cpu_cores_used: number | null;
};

export type Benchmark = {
  run_id: string;
  results: BenchmarkItem[];
  spark_vs_pandas_speedup: number | null;
  spark_vs_joblib_speedup: number | null;
};

// A single fetch wrapper. Returns null on any error/404 so callers can render an
// empty state instead of crashing the page.
export async function apiGet<T>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export async function apiPost<T>(path: string, body: unknown): Promise<T | null> {
  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      cache: "no-store",
    });
    if (!res.ok) return null;
    return (await res.json()) as T;
  } catch {
    return null;
  }
}

export const RISK_COLOR: Record<string, string> = {
  low: "#16a34a",
  medium: "#d97706",
  high: "#dc2626",
};

export const ENGINE_COLOR: Record<string, string> = {
  pyspark: "#4f46e5",
  pandas: "#0891b2",
  joblib: "#16a34a",
};
