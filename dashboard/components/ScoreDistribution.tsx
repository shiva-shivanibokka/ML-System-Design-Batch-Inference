"use client";

import { useEffect, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { apiGet, BatchRun, ScoreDistribution as Dist } from "@/lib/api";
import { Card, Empty, fmt } from "@/components/ui";

function riskColor(mid: number) {
  return mid >= 0.7 ? "#dc2626" : mid >= 0.4 ? "#d97706" : "#16a34a";
}

export default function ScoreDistribution() {
  const [runId, setRunId] = useState("");
  const [dist, setDist] = useState<Dist | null>(null);
  const [loading, setLoading] = useState(true);

  async function load(id?: string) {
    setLoading(true);
    // Default to the latest completed run if no id is given.
    let target = id?.trim();
    if (!target) {
      const latest = await apiGet<BatchRun>("/batch-runs/latest");
      target = latest?.run_id;
    }
    setDist(target ? await apiGet<Dist>(`/batch-runs/${target}/distribution`) : null);
    setLoading(false);
  }

  useEffect(() => {
    load();
  }, []);

  const chart =
    dist?.bins.map((b) => ({
      range: `${b.bin_lower.toFixed(1)}–${b.bin_upper.toFixed(1)}`,
      fraction: b.fraction,
      count: b.count,
      color: riskColor((b.bin_lower + b.bin_upper) / 2),
    })) ?? [];

  const total = dist?.bins.reduce((s, b) => s + b.count, 0) ?? 0;
  const tier = (lo: number, hi: number) =>
    dist?.bins.filter((b) => b.bin_upper > lo && b.bin_upper <= hi).reduce((s, b) => s + b.count, 0) ?? 0;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={runId}
          onChange={(e) => setRunId(e.target.value)}
          placeholder="run-id (blank = latest)"
          className="rounded-md border px-3 py-1.5 text-sm"
          style={{ background: "var(--card)", borderColor: "var(--border)" }}
        />
        <button
          onClick={() => load(runId)}
          className="rounded-md px-3 py-1.5 text-sm font-medium text-white"
          style={{ background: "#4f46e5" }}
        >
          Load
        </button>
      </div>

      {loading ? (
        <Empty msg="Loading…" />
      ) : !dist ? (
        <Empty msg="No score distribution found for that run." />
      ) : (
        <>
          <Card title={`Churn score distribution · ${dist.run_id}`}>
            <ResponsiveContainer width="100%" height={320}>
              <BarChart data={chart}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
                <XAxis dataKey="range" fontSize={11} angle={-25} textAnchor="end" height={50} />
                <YAxis fontSize={11} />
                <Tooltip formatter={(v: number, n) => (n === "fraction" ? `${(v * 100).toFixed(1)}%` : v)} />
                <Bar dataKey="fraction" name="fraction">
                  {chart.map((d, i) => (
                    <Cell key={i} fill={d.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
            <div className="mt-2 flex gap-4 text-xs" style={{ color: "var(--muted)" }}>
              <span>🟢 Low (&lt;0.4)</span>
              <span>🟠 Medium (0.4–0.7)</span>
              <span>🔴 High (&gt;0.7)</span>
            </div>
          </Card>

          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Card><div className="text-xs" style={{ color: "var(--muted)" }}>Low risk</div><div className="text-xl font-semibold">{fmt(tier(0, 0.4))}</div></Card>
            <Card><div className="text-xs" style={{ color: "var(--muted)" }}>Medium risk</div><div className="text-xl font-semibold">{fmt(tier(0.4, 0.7))}</div></Card>
            <Card><div className="text-xs" style={{ color: "var(--muted)" }}>High risk</div><div className="text-xl font-semibold">{fmt(tier(0.7, 1.0))}</div></Card>
            <Card><div className="text-xs" style={{ color: "var(--muted)" }}>Total scored</div><div className="text-xl font-semibold">{fmt(total)}</div></Card>
          </div>
        </>
      )}
    </div>
  );
}
