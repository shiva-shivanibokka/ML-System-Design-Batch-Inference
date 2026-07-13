"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiGet, BatchRun, Stats } from "@/lib/api";
import { Card, Empty, StatTile, StatusBadge, fmt } from "@/components/ui";

type ListResp = { total: number; items: BatchRun[] };

export default function BatchRuns() {
  const [runs, setRuns] = useState<BatchRun[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  async function load() {
    setLoading(true);
    const [list, s] = await Promise.all([
      apiGet<ListResp>("/batch-runs?size=20"),
      apiGet<Stats>("/stats"),
    ]);
    setRuns(list?.items ?? []);
    setStats(s);
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  if (loading) return <Empty msg="Loading…" />;
  if (!runs.length)
    return <Empty msg="No batch runs yet. Trigger the pipeline, then refresh." />;

  // Oldest → newest for trend charts.
  const trend = [...runs]
    .filter((r) => r.score_mean !== null)
    .reverse()
    .map((r) => ({
      date: r.started_at?.slice(0, 10),
      mean: r.score_mean,
      p50: r.score_p50,
      psi: r.psi_vs_previous,
    }));

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <StatTile label="Total runs" value={fmt(stats?.total_batch_runs)} />
        <StatTile label="Predictions" value={fmt(stats?.total_predictions)} />
        <StatTile
          label="Avg Spark (s)"
          value={fmt(stats?.avg_spark_duration_secs, 1)}
        />
        <StatTile label="Avg rec/s" value={fmt(stats?.avg_records_per_second)} />
      </div>

      <Card title="Pipeline executions">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead style={{ color: "var(--muted)" }}>
              <tr className="border-b" style={{ borderColor: "var(--border)" }}>
                <th className="py-2 pr-4">Run ID</th>
                <th className="py-2 pr-4">Status</th>
                <th className="py-2 pr-4">Started</th>
                <th className="py-2 pr-4 text-right">Scored</th>
                <th className="py-2 pr-4 text-right">Spark (s)</th>
                <th className="py-2 pr-4 text-right">PSI</th>
                <th className="py-2 pr-4 text-right">rec/s</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id} className="border-b" style={{ borderColor: "var(--border)" }}>
                  <td className="py-2 pr-4 font-mono text-xs">{r.run_id}</td>
                  <td className="py-2 pr-4"><StatusBadge status={r.status} /></td>
                  <td className="py-2 pr-4">{r.started_at?.replace("T", " ").slice(0, 16)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.records_scored)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.spark_duration_secs, 1)}</td>
                  <td className="py-2 pr-4 text-right tabular-nums">
                    {r.psi_vs_previous === null ? "—" : (
                      <span style={{ color: r.drift_flagged ? "#dc2626" : undefined }}>
                        {r.psi_vs_previous.toFixed(3)}
                      </span>
                    )}
                  </td>
                  <td className="py-2 pr-4 text-right tabular-nums">{fmt(r.records_per_second)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Churn score trend">
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" fontSize={11} />
              <YAxis fontSize={11} domain={[0, "auto"]} />
              <Tooltip />
              <Line type="monotone" dataKey="mean" name="Mean" stroke="#4f46e5" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="p50" name="P50" stroke="#0891b2" strokeWidth={2} strokeDasharray="4 4" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </Card>

        <Card title="PSI drift (Basel II: >0.2 investigate)">
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey="date" fontSize={11} />
              <YAxis fontSize={11} />
              <Tooltip />
              <ReferenceLine y={0.2} stroke="#dc2626" strokeDasharray="4 4" />
              <ReferenceLine y={0.1} stroke="#d97706" strokeDasharray="4 4" />
              <Bar dataKey="psi" name="PSI">
                {trend.map((d, i) => (
                  <Cell
                    key={i}
                    fill={(d.psi ?? 0) > 0.2 ? "#dc2626" : (d.psi ?? 0) > 0.1 ? "#d97706" : "#16a34a"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </div>
    </div>
  );
}
