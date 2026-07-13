"use client";

import { useState } from "react";
import { apiGet, apiPost, CustomerScore } from "@/lib/api";
import { Card, Empty, RiskBadge } from "@/components/ui";

type Bulk = { found: CustomerScore[]; not_found: string[] };

export default function CustomerLookup() {
  const [id, setId] = useState("");
  const [single, setSingle] = useState<CustomerScore | null>(null);
  const [singleMsg, setSingleMsg] = useState("");
  const [bulkText, setBulkText] = useState("");
  const [bulk, setBulk] = useState<Bulk | null>(null);

  async function lookupSingle() {
    const cid = id.trim();
    if (!cid) return;
    setSingle(null);
    setSingleMsg("Looking up…");
    const res = await apiGet<CustomerScore>(`/score/${encodeURIComponent(cid)}`);
    if (res) {
      setSingle(res);
      setSingleMsg("");
    } else {
      setSingleMsg(`No score found for "${cid}".`);
    }
  }

  async function lookupBulk() {
    const ids = bulkText.split("\n").map((s) => s.trim()).filter(Boolean).slice(0, 500);
    if (!ids.length) return;
    setBulk(await apiPost<Bulk>("/scores/bulk", { customer_ids: ids }));
  }

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card title="Single customer">
        <div className="flex gap-2">
          <input
            value={id}
            onChange={(e) => setId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && lookupSingle()}
            placeholder="customer_id (KKBox msno)"
            className="flex-1 rounded-md border px-3 py-1.5 text-sm"
            style={{ background: "var(--card)", borderColor: "var(--border)" }}
          />
          <button
            onClick={lookupSingle}
            className="rounded-md px-3 py-1.5 text-sm font-medium text-white"
            style={{ background: "#4f46e5" }}
          >
            Look up
          </button>
        </div>

        {singleMsg && <p className="mt-3 text-sm" style={{ color: "var(--muted)" }}>{singleMsg}</p>}
        {single && (
          <dl className="mt-4 space-y-2 text-sm">
            <Row k="Risk" v={<RiskBadge tier={single.risk_tier} />} />
            <Row k="Churn probability" v={`${(single.churn_probability * 100).toFixed(1)}%`} />
            <Row k="Decile" v={`${single.churn_decile} / 10`} />
            <Row k="Model" v={single.model_version} />
            <Row k="Scored at" v={single.scored_at?.replace("T", " ").slice(0, 19)} />
            <Row k="Run" v={<span className="font-mono text-xs">{single.run_id}</span>} />
          </dl>
        )}
      </Card>

      <Card title="Bulk lookup (up to 500 IDs, one per line)">
        <textarea
          value={bulkText}
          onChange={(e) => setBulkText(e.target.value)}
          rows={6}
          placeholder={"id-1\nid-2\nid-3"}
          className="w-full rounded-md border px-3 py-2 font-mono text-xs"
          style={{ background: "var(--card)", borderColor: "var(--border)" }}
        />
        <button
          onClick={lookupBulk}
          className="mt-2 rounded-md px-3 py-1.5 text-sm font-medium text-white"
          style={{ background: "#4f46e5" }}
        >
          Look up all
        </button>

        {bulk && (
          <div className="mt-4 max-h-72 overflow-auto">
            {bulk.found.length === 0 ? (
              <Empty msg="No matches found." />
            ) : (
              <table className="w-full text-left text-xs">
                <thead style={{ color: "var(--muted)" }}>
                  <tr>
                    <th className="py-1 pr-2">Customer</th>
                    <th className="py-1 pr-2">Prob</th>
                    <th className="py-1 pr-2">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {bulk.found.map((r) => (
                    <tr key={r.customer_id} className="border-t" style={{ borderColor: "var(--border)" }}>
                      <td className="py-1 pr-2 font-mono">{r.customer_id}</td>
                      <td className="py-1 pr-2 tabular-nums">{(r.churn_probability * 100).toFixed(1)}%</td>
                      <td className="py-1 pr-2"><RiskBadge tier={r.risk_tier} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
            {bulk.not_found.length > 0 && (
              <p className="mt-2 text-xs" style={{ color: "var(--muted)" }}>
                Not found: {bulk.not_found.length}
              </p>
            )}
          </div>
        )}
      </Card>
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between border-b pb-1" style={{ borderColor: "var(--border)" }}>
      <dt style={{ color: "var(--muted)" }}>{k}</dt>
      <dd className="font-medium">{v}</dd>
    </div>
  );
}
