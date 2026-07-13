"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiGet, Benchmark as Bench, ENGINE_COLOR } from "@/lib/api";
import { Card, Empty } from "@/components/ui";

const ENGINES = ["pandas", "joblib", "pyspark"] as const;

export default function Benchmark() {
  const [data, setData] = useState<Bench | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    (async () => {
      setData(await apiGet<Bench>("/benchmark"));
      setLoading(false);
    })();
  }, []);

  if (loading) return <Empty msg="Loading…" />;
  if (!data) return <Empty msg="No benchmark results yet. Run the pipeline's benchmark task." />;

  // Pivot to one row per sample size, one column per engine.
  const sizes = Array.from(new Set(data.results.map((r) => r.sample_size))).sort((a, b) => a - b);
  const byThroughput = sizes.map((size) => {
    const row: Record<string, number | string> = { size: size.toLocaleString() };
    for (const e of ENGINES) {
      const hit = data.results.find((r) => r.engine === e && r.sample_size === size);
      if (hit) row[e] = hit.records_per_second;
    }
    return row;
  });
  const byDuration = sizes.map((size) => {
    const row: Record<string, number | string> = { size };
    for (const e of ENGINES) {
      const hit = data.results.find((r) => r.engine === e && r.sample_size === size);
      if (hit) row[e] = hit.duration_secs;
    }
    return row;
  });

  return (
    <div className="space-y-4">
      <Card>
        <p className="text-sm" style={{ color: "var(--muted)" }}>
          When does each engine win? PySpark carries JVM startup overhead, so at these
          single-machine sizes pandas/joblib lead — PySpark&apos;s advantage is at cluster
          scale. Deployed batches use the right-sized engine for the data volume.
          {data.spark_vs_pandas_speedup !== null && (
            <> At the largest size, PySpark is {data.spark_vs_pandas_speedup}× pandas.</>
          )}
        </p>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Throughput (records/sec)">
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={byThroughput}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="size" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <Legend />
              {ENGINES.map((e) => (
                <Bar key={e} dataKey={e} fill={ENGINE_COLOR[e]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </Card>

        <Card title="Duration (s, log scale)">
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={byDuration}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="size" fontSize={11} scale="log" domain={["auto", "auto"]} type="number" />
              <YAxis fontSize={11} scale="log" domain={["auto", "auto"]} />
              <Tooltip />
              <Legend />
              {ENGINES.map((e) => (
                <Line key={e} type="monotone" dataKey={e} stroke={ENGINE_COLOR[e]} strokeWidth={2} dot />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
